import argparse
import copy
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.CelebA_dataset import CelebADataset
from evaluation import (
    load_child_models_X2C,
    load_child_models_DivEns,
    load_independent_models,
    load_child_models_Partial,
    load_child_models_Adapter,
    load_child_models_flex,
)
from models import LoraEnsemble, LoraEnsembleshared, LoraEnsemblePshared

N_CLASSES = 256
N_BITS = 8

def get_balanced_attr_names(attr_path: str) -> list[str]:
    """Re-run the setup balance-score logic → 8 most-balanced CelebA attribute names."""
    df = pd.read_csv(attr_path, delim_whitespace=True, header=1)
    df_bin = (df + 1) // 2
    scores  = (df_bin.mean() - 0.5).abs()
    return scores.nsmallest(N_BITS).index.tolist()    # stable order


def decode_bit(class_array: np.ndarray, pos: int) -> np.ndarray:
    """
    Decode the bit at position `pos` (0 = MSB) from an 8-bit class integer.
    binary_vector_to_decimal([a0,a1,...,a7]) = int(''.join(str(v) for v in attrs), 2)
    → attribute pos is at bit (7-pos) of the integer.
    """
    return (class_array >> (N_BITS - 1 - pos)) & 1


def smiling_classes(smiling_pos: int) -> np.ndarray:
    """Return the 128 class indices where the Smiling bit is 1."""
    return np.array([c for c in range(N_CLASSES) if decode_bit(np.array([c]), smiling_pos)[0]])

def build_fairness_loader(
    pkl_path: str,
    attr_path: str,
    protected_attr: str,
    target_attr: str,
    batch_size: int,
) -> tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Returns (loader, protected_labels, target_labels).
    protected_labels / target_labels are aligned with loader (no shuffle).
    """
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    df_attr = pd.read_csv(attr_path, delim_whitespace=True, header=1)
    df_bin  = (df_attr + 1) // 2  # –1/+1 → 0/1

    for col in (protected_attr, target_attr):
        if col not in df_bin.columns:
            raise ValueError(f"Attribute '{col}' not found in {attr_path}. "
                             f"Available: {sorted(df_bin.columns)}")

    protected, target = [], []
    for sample in data_list:
        fname = os.path.basename(sample["image_path"])
        row   = df_bin.loc[fname]
        protected.append(int(row[protected_attr]))
        target.append(int(row[target_attr]))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    dataset = CelebADataset(data_list=data_list, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, np.array(protected), np.array(target)

def _call_model(model: nn.Module, imgs: torch.Tensor):
    """
    Unified forward call.  All child models return (y_logits, c_list) but
    _ChildForSHAP requires an explicit `is_training` positional argument
    while _ChildModel / SingleE2EBranch do not (or have a default).
    """
    try:
        return model(imgs, False)
    except TypeError:
        return model(imgs)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    target_pos: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        y_hat  : [N]    predicted target attribute (0/1) from argmax class
        c_hat  : [N, K] sigmoid of concept logits
    """
    model.eval()
    y_hat_list, c_hat_list = [], []

    for batch in tqdm(loader, leave=False, desc="  inference"):
        imgs = batch["img"].to(device)
        y_logits, c_list = _call_model(model, imgs)

        # predicted class → decode target bit
        pred_class  = y_logits.argmax(dim=1).cpu().numpy()
        y_hat_batch = decode_bit(pred_class, target_pos)

        # concept predictions → sigmoid → [B, K]
        c_tensor = torch.cat(c_list, dim=1)          # [B, K]
        c_sigmoid = torch.sigmoid(c_tensor).cpu().numpy()

        y_hat_list.append(y_hat_batch)
        c_hat_list.append(c_sigmoid)

    return np.concatenate(y_hat_list), np.concatenate(c_hat_list)

def demographic_parity_gap(y_hat: np.ndarray, s: np.ndarray) -> float:
    """P(Ŷ=1|S=1) – P(Ŷ=1|S=0)"""
    if s.sum() == 0 or (1 - s).sum() == 0:
        return float("nan")
    return y_hat[s == 1].mean() - y_hat[s == 0].mean()


def equal_opportunity_gap(y_hat: np.ndarray, y_true: np.ndarray, s: np.ndarray) -> float:
    """P(Ŷ=1|Y=1,S=1) – P(Ŷ=1|Y=1,S=0)"""
    m1 = (s == 1) & (y_true == 1)
    m0 = (s == 0) & (y_true == 1)
    if m1.sum() == 0 or m0.sum() == 0:
        return float("nan")
    return y_hat[m1].mean() - y_hat[m0].mean()


def raw_concept_gaps(c_hat: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E[Ĉk|S=1] – E[Ĉk|S=0] for each concept k.  Shape: [K]"""
    return c_hat[s == 1].mean(0) - c_hat[s == 0].mean(0)


def effective_head_weights(model: nn.Module, target_cls: np.ndarray) -> np.ndarray | None:
    """
    Effective linear-head weight for the target attribute:
        w_eff[k] = mean_{c∈target=1}(W[c,k]) – mean_{c∈target=0}(W[c,k])
    Returns shape [K], or None if the head is not a simple linear layer.
    """
    c_to_y = getattr(model, "model_c_to_y", None)
    if c_to_y is None:
        return None
    linear = getattr(c_to_y, "linear", None)
    if linear is None:
        return None
    W = linear.weight.detach().cpu().numpy()       # [N_CLASSES, K]
    non_target = np.setdiff1d(np.arange(N_CLASSES), target_cls)
    return W[target_cls].mean(0) - W[non_target].mean(0)   # [K]


def weighted_concept_gaps(
    c_hat: np.ndarray, s: np.ndarray, w_eff: np.ndarray
) -> np.ndarray:
    """wk · (E[Ĉk|S=1] – E[Ĉk|S=0])  Shape: [K]"""
    return w_eff * raw_concept_gaps(c_hat, s)

def _hist(ax, values, title, xlabel, color):
    valid = [v for v in values if not np.isnan(v)]
    n_bins = max(5, len(valid) // 3)
    ax.hist(valid, bins=n_bins, color=color, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("# models", fontsize=9)
    mu = np.nanmean(values)
    ax.axvline(mu, color="navy", linestyle=":", linewidth=1.2,
               label=f"mean={mu:+.3f}")
    ax.legend(fontsize=8)


def plot_outcome_fairness(
    dp_gaps: np.ndarray,
    eo_gaps: np.ndarray,
    title_tag: str,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _hist(axes[0], dp_gaps, f"Demographic Parity Gap\n{title_tag}",
          "DP gap  (S=1 minus S=0)", "steelblue")
    _hist(axes[1], eo_gaps, f"Equal Opportunity Gap\n{title_tag}",
          "EO gap  (S=1 minus S=0)", "salmon")
    fig.tight_layout()
    path = os.path.join(out_dir, "fairness_outcome.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.close(fig)


def plot_concept_gaps(
    all_gaps: list[np.ndarray],
    concept_names: list[str],
    title_tag: str,
    out_dir: str,
    weighted: bool = False,
) -> None:
    K   = len(concept_names)
    mat = np.stack(all_gaps)              # [n_models, K]
    tag = "weighted" if weighted else "raw"

    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 4), sharey=False)
    if K == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, K))
    for k, ax in enumerate(axes):
        _hist(ax, mat[:, k].tolist(),
              concept_names[k], f"{tag} gap (S=1–S=0)", colors[k])

    fig.suptitle(f"Per-concept fairness gap  ({tag})\n{title_tag}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"fairness_concept_{tag}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.close(fig)

def _set_lora_adapter(ensemble: nn.Module, i: int) -> None:
    """Activate the correct LoRA adapter(s) for child i."""
    if isinstance(ensemble, LoraEnsemblePshared):
        ensemble.lora_models.set_adapter([f"lora_{i}", "lora_shared"])
    elif isinstance(ensemble, LoraEnsembleshared):
        ensemble.lora_models.set_adapter("lora_shared")
    else:  # LoraEnsemble
        ensemble.lora_models.set_adapter(f"lora_{i}")


class _LoraChildModel(nn.Module):
    """Per-child wrapper around a LoraEnsemble for fairness evaluation.

    Exposes:
      • forward(x) → (y_logits, c_list)
      • model_c_to_y  – the MLP head (has .linear for effective_head_weights)
    """

    def __init__(self, ensemble: nn.Module, child_idx: int) -> None:
        super().__init__()
        self._ensemble = ensemble
        self._i = child_idx
        # Expose so that effective_head_weights() can access .linear
        self.model_c_to_y = ensemble.final_heads[child_idx]["classifier"]

    def forward(self, x, is_training: bool = False):
        _set_lora_adapter(self._ensemble, self._i)
        c_list = self._ensemble.model_x_to_c(x, self._i)
        c = torch.cat(c_list, dim=1)
        y = self._ensemble.model_c_to_y(c, self._i)
        return y, c_list

def load_child_models(args, device, n_concepts: int) -> list[nn.Module]:
    # Support log_dir pointing directly to the .pth file
    if args.log_dir.endswith(".pth"):
        model_path = args.log_dir
    else:
        model_path = os.path.join(args.log_dir, "best_model.pth")

    if args.exp == "Lora":
        mask = args.share_mask
        if args.encoder not in ("vit", "medical_vit"):
            raise ValueError("--exp Lora requires --encoder vit or medical_vit.")
        if len(mask) != 12 or any(ch not in "01" for ch in mask):
            raise ValueError("--exp Lora requires a 12-bit --share_mask, e.g. 000000000000.")
        lora_kwargs = dict(
            num_models=args.num_models,
            num_classes=N_CLASSES,
            n_attributes=n_concepts,
            encoder=args.encoder,
            expand_dim=args.expand_dim,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        if mask == "000000000000":
            ensemble = LoraEnsemble(**lora_kwargs, lora_block_mask=mask)
        elif mask == "111111111111":
            ensemble = LoraEnsembleshared(**lora_kwargs, lora_block_mask=mask)
        else:
            ensemble = LoraEnsemblePshared(**lora_kwargs, lora_block_mask=mask)

        ensemble.load_state_dict(
            torch.load(model_path, map_location=device), strict=True
        )
        ensemble.to(device).eval()
        return [_LoraChildModel(ensemble, i) for i in range(args.num_models)]

    if args.exp == "X2C":
        return load_child_models_X2C(model_path, args, device, n_concepts, N_CLASSES)
    elif args.exp == "DivEns":
        return load_child_models_DivEns(model_path, args.num_models, device,
                                     args, n_concepts, N_CLASSES)
    elif args.exp == "random":
        return load_independent_models(model_path, args, device, n_concepts, N_CLASSES)
    elif args.exp == "PartialX2C":
        return load_child_models_Partial(
            torch.load(model_path, map_location=device), args=args, device=device
        )
    elif args.exp == "ConvParX2C":
        return load_child_models_Adapter(model_path, args, device, n_concepts, N_CLASSES)
    elif args.exp == "ConvAda":
        if args.encoder != "resnet18":
            raise ValueError("--exp ConvAda requires --encoder resnet18.")
        proxy = type("A", (), {
            "num_models": args.num_models,
            "encoder": args.encoder,
            "expand_dim": args.expand_dim,
            "bottleneck_dim": args.bottleneck_dim,
            "share_mask": args.share_mask,
        })()
        return load_child_models_flex(model_path, proxy, device, n_concepts, N_CLASSES)
    else:
        raise ValueError(f"Unsupported --exp: {args.exp}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fairness evaluation across a Rashomon set (CelebA)"
    )
    parser.add_argument("--log_dir",   required=True,
                        help="Directory with best_model.pth, or path to the .pth file directly")
    parser.add_argument("--data_dir",  required=True,
                        help="Dir containing test.pkl and celeba/list_attr_celeba.txt")
    parser.add_argument("--exp",       default="Lora",
                        choices=["DivEns", "random", "ConvAda", "Lora"])
    parser.add_argument("--num_models",   type=int, default=10)
    parser.add_argument("--n_attributes", type=int, default=6)
    parser.add_argument("--expand_dim",   type=int, default=0)
    parser.add_argument("--bottleneck_dim", type=int, default=64,
                        help="Adapter bottleneck dimension for ConvAda checkpoints.")
    parser.add_argument("--n_class_attr", type=int, default=2)
    parser.add_argument("--encoder",      default="vit")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--share_mask",   default="000000000000")
    parser.add_argument("--lora_r",       type=int, default=8)
    parser.add_argument("--lora_alpha",   type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--out_dir",      default=None,
                        help="Output directory for plots/CSVs "
                             "(default: fairness_results/{target}_vs_{protected})")
    parser.add_argument("--protected_attr", default="Young",
                        help="Protected attribute name as in list_attr_celeba.txt")
    parser.add_argument("--target_attr",    default="Heavy_Makeup",
                        help="Target prediction attribute name")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        "fairness_results",
        f"{args.target_attr}_vs_{args.protected_attr}",
    )
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attr_path = os.path.join(args.data_dir, "list_attr_celeba.txt")
    balanced_attrs = get_balanced_attr_names(attr_path)
    concept_names  = balanced_attrs[:args.n_attributes]

    print(f"8 balanced attrs : {balanced_attrs}")
    print(f"Concept names    : {concept_names}")

    if args.target_attr not in balanced_attrs:
        raise RuntimeError(
            f"'{args.target_attr}' is not among the 8 balanced attrs selected "
            f"during setup: {balanced_attrs}.\n"
            f"Try --target_attr with one of those names."
        )
    target_pos    = balanced_attrs.index(args.target_attr)
    target_cls    = smiling_classes(target_pos)          # 128 class indices
    print(f"'{args.target_attr}' → bit position {target_pos}, "
          f"{len(target_cls)}/256 classes")

    test_pkl = os.path.join(args.data_dir, "test.pkl")
    loader, s_labels, y_true = build_fairness_loader(
        test_pkl, attr_path, args.protected_attr, args.target_attr, args.batch_size
    )
    N = len(s_labels)
    print(f"\nTest set : {N} samples | "
          f"{args.protected_attr}=1: {s_labels.mean():.1%} | "
          f"{args.target_attr}=1: {y_true.mean():.1%}")

    child_models = load_child_models(args, device, args.n_attributes)
    print(f"Loaded {len(child_models)} child models.\n")

    dp_gaps, eo_gaps         = [], []
    all_raw_gaps             = []
    all_weighted_gaps        = []

    title_tag = (f"{args.exp} | target={args.target_attr} "
                 f"| protected={args.protected_attr}")

    for i, model in enumerate(child_models):
        model.to(device).eval()
        print(f"Model {i:>2} …", end=" ", flush=True)

        y_hat, c_hat = run_inference(model, loader, target_pos, device)

        dp = demographic_parity_gap(y_hat, s_labels)
        eo = equal_opportunity_gap(y_hat, y_true, s_labels)
        dp_gaps.append(dp)
        eo_gaps.append(eo)
        print(f"DP={dp:+.4f}  EO={eo:+.4f}")

        # raw concept gaps
        all_raw_gaps.append(raw_concept_gaps(c_hat, s_labels))

        # linear-head weighted gaps (only if head is a simple linear layer)
        w_eff = effective_head_weights(model, target_cls)
        if w_eff is not None:
            all_weighted_gaps.append(weighted_concept_gaps(c_hat, s_labels, w_eff))

    dp_gaps = np.array(dp_gaps)
    eo_gaps = np.array(eo_gaps)

    print(f"\nDP gap:  mean={np.nanmean(dp_gaps):+.4f}  std={np.nanstd(dp_gaps):.4f}")
    print(f"EO gap:  mean={np.nanmean(eo_gaps):+.4f}  std={np.nanstd(eo_gaps):.4f}")

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_outcome_fairness(dp_gaps, eo_gaps, title_tag, out_dir)
    plot_concept_gaps(all_raw_gaps, concept_names, title_tag, out_dir, weighted=False)
    if all_weighted_gaps:
        plot_concept_gaps(all_weighted_gaps, concept_names, title_tag, out_dir, weighted=True)
    else:
        print("  (weighted concept gaps skipped – no simple linear head found)")

    pd.DataFrame({"dp_gap": dp_gaps, "eo_gap": eo_gaps}).to_csv(
        os.path.join(out_dir, "fairness_outcome.csv"), index=False
    )
    pd.DataFrame(np.stack(all_raw_gaps), columns=concept_names).to_csv(
        os.path.join(out_dir, "fairness_concept_raw.csv"), index=False
    )
    if all_weighted_gaps:
        pd.DataFrame(np.stack(all_weighted_gaps), columns=concept_names).to_csv(
            os.path.join(out_dir, "fairness_concept_weighted.csv"), index=False
        )

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
