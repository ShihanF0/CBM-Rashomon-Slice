import os
import sys
import math
import copy
import json
import csv
import argparse
import re
import pickle
import hashlib
import time
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

# Project imports (expected to exist in your repo)
from datasets.CUB_dataset import load_CUB_data
from datasets.Awa2_dataset import load_awa2_data
from datasets.CelebA_dataset import load_celeba_data
from datasets.HAM10000_dataset import load_ham10000_data

from config import (
    BASE_DIR,
    CUB_N_CLASSES,
    AWA2_N_CLASSES,
    CIFAR10_N_CLASSES,
    CELEBA_N_CLASSES,
    HAM10000_N_CLASSES,
)

from models import (
    SingleE2EBranch,
    EnsembleWithPartialSharing,
    ConvParEnsemble,
    SConvParEnsemble,
    LoraEnsemble,
    LoraEnsemblePshared,
    LoraEnsembleshared,
)

N_CLASSES: int = -1
CONCEPT_DIM: int = -1

def _sanitize_tag(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s.strip("_") or "run"


def _parse_class_idxs(s: str) -> Optional[set[int]]:
    """Parse class idx filter.

    - "-1" (or empty) => None (no filter)
    - "0,1,5" or "0 1 5" => set{0,1,5}
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s == "-1":
        return None
    parts = re.split(r"[,\s]+", s)
    out: List[int] = []
    for p in parts:
        if p == "":
            continue
        try:
            out.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid token '{p}' in --class_idxs='{s}'") from e
    return set(out) if out else None


def _parse_class_sweep(spec: str) -> List[str]:
    """Parse a sweep list for class_idxs.

    Format:
      - empty -> []
      - parts separated by ';' (recommended) or '|' e.g. "-1;0;1;2;3;4;5;6"
      - each part is itself a valid class_idxs string: "-1" or "0" or "0,1,2"
    """
    s = str(spec or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[;|]", s) if p.strip() != ""]
    out: List[str] = []
    for p in parts:
        out.append("-1" if p.lower() == "all" else p)
    return out


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _append_summary_csv(csv_path: str, row: Dict[str, object]) -> None:
    """Append a single row to a CSV. Creates file with header if missing."""
    _ensure_dir(os.path.dirname(csv_path) or ".")
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _load_runs_from_json(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "runs" in obj:
        runs = obj["runs"]
    else:
        runs = obj
    if not isinstance(runs, list):
        raise ValueError("--runs_json must contain a list or a dict with key 'runs'")
    out: List[Dict[str, object]] = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ValueError(f"Run entry #{i} must be an object")
        out.append(r)
    return out

def _load_split_loaders(args):
    root = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(BASE_DIR, args.data_dir)
    train_pkl = os.path.join(root, "train.pkl")
    val_pkl = os.path.join(root, "val.pkl")
    test_pkl = os.path.join(root, "test.pkl")

    def _parse_eval_splits(s: str) -> List[str]:
        s = (s or "test").strip().lower()
        if s in {"test", "val", "train"}:
            return [s]
        parts = [p.strip().lower() for p in re.split(r"[,\s]+", s) if p.strip() != ""]
        if not parts:
            return ["test"]
        allowed = {"train", "val", "test"}
        bad = [p for p in parts if p not in allowed]
        if bad:
            raise ValueError(f"Invalid --eval_splits={s!r}. Allowed: train,val,test (comma-separated).")
        # keep order but de-dup
        out = []
        seen = set()
        for p in parts:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def _combined_eval_pkl() -> str:
        splits = _parse_eval_splits(getattr(args, "eval_splits", "test"))
        if splits == ["test"]:
            return test_pkl
        mapping = {"train": train_pkl, "val": val_pkl, "test": test_pkl}
        lists: List[object] = []
        for sp in splits:
            path = mapping[sp]
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, list):
                raise TypeError(f"{sp}.pkl did not contain a list (got {type(obj)}): {path}")
            lists.extend(obj)
        # Write combined list to a temp pkl so we can reuse existing loaders.
        key = f"{os.path.abspath(root)}|{','.join(splits)}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
        out = os.path.join(tempfile.gettempdir(), f"combined_{Path(root).name}_{h}_{os.getpid()}.pkl")
        with open(out, "wb") as f:
            pickle.dump(lists, f)
        print(f"[Data] Using combined eval splits {splits} -> {out} (n={len(lists)})")
        return out

    eval_pkl = _combined_eval_pkl()

    if args.dataname == "CUB":
        return (
            load_CUB_data(train_pkl, batch_size=args.batch_size, is_training=False),
            load_CUB_data(val_pkl, batch_size=args.batch_size, is_training=False),
            load_CUB_data(eval_pkl, batch_size=args.batch_size, is_training=False),
        )
    if args.dataname == "Awa2":
        return (
            load_awa2_data(train_pkl, batch_size=args.batch_size, is_training=False),
            load_awa2_data(val_pkl, batch_size=args.batch_size, is_training=False),
            load_awa2_data(eval_pkl, batch_size=args.batch_size, is_training=False),
        )
    if args.dataname == "cifar10":
        # kept as-is from your original script
        return (
            load_awa2_data(train_pkl, batch_size=args.batch_size, is_training=False),
            load_awa2_data(val_pkl, batch_size=args.batch_size, is_training=False),
            load_awa2_data(eval_pkl, batch_size=args.batch_size, is_training=False),
        )
    if args.dataname == "CelebA":
        return (
            load_celeba_data(train_pkl, batch_size=args.batch_size, is_training=False),
            load_celeba_data(val_pkl, batch_size=args.batch_size, is_training=False),
            load_celeba_data(eval_pkl, batch_size=args.batch_size, is_training=False),
        )
    if args.dataname == "HAM10000":
        return (
            load_ham10000_data(train_pkl, batch_size=args.batch_size, is_training=False, encoder=getattr(args, "encoder", "resnet18")),
            load_ham10000_data(val_pkl, batch_size=args.batch_size, is_training=False, encoder=getattr(args, "encoder", "resnet18")),
            load_ham10000_data(eval_pkl, batch_size=args.batch_size, is_training=False, encoder=getattr(args, "encoder", "resnet18")),
        )
    raise ValueError(f"Unknown dataname: {args.dataname}")


# =======================================================
# Child model extraction (copied/adapted from your original eval)
# =======================================================
def load_child_models_X2C(ensemble_model_path, args, device, num_concepts, num_classes):
    """Standard X2C ensemble saved with `branches.{i}.` prefixes."""
    ensemble_state_dict = torch.load(ensemble_model_path, map_location="cpu")

    child_models = []
    num_models = args.num_models
    print(f"[Extract] X2C: extracting {num_models} child models from {ensemble_model_path} ...")

    for i in range(num_models):
        child_model = SingleE2EBranch(
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=num_classes,
            use_aux=True,
            n_attributes=num_concepts,
            expand_dim=args.expand_dim,
            encoder=args.encoder,
        )

        child_state_dict = {}
        prefix = f"branches.{i}."
        for key, value in ensemble_state_dict.items():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "", 1)
                child_state_dict[new_key] = value

        if not child_state_dict:
            print(f"[Extract] Warning: no weights for child {i} using prefix '{prefix}'. Skipping.")
            continue

        child_model.load_state_dict(child_state_dict, strict=False)
        child_model.to(device).eval()
        child_models.append(child_model)

    if len(child_models) == 0:
        raise RuntimeError("No child models were extracted. Check `args.num_models` and checkpoint structure.")
    return child_models


def load_child_models_DivEns(ensemble_model_path, num_models, device, args, num_concepts, num_classes):
    """DivEns: shared X->C and separate C->Y heads under `branches_c_to_y.{i}.model_c_to_y.`."""
    print(f"[Extract] DivEns: loading from {ensemble_model_path}...")
    ensemble_state_dict = torch.load(ensemble_model_path, map_location="cpu")
    child_models = []

    # shared X->C
    model_x_to_c_state_dict = {}
    shared_prefix = "model_x_to_c."
    for key, value in ensemble_state_dict.items():
        if key.startswith(shared_prefix):
            new_key = key.replace(shared_prefix, "")
            model_x_to_c_state_dict[new_key] = value
    if not model_x_to_c_state_dict:
        raise ValueError("No shared X->C weights found. Check checkpoint keys.")

    for i in range(num_models):
        child_model = SingleE2EBranch(
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=num_classes,
            use_aux=True,
            n_attributes=num_concepts,
            expand_dim=args.expand_dim,
            encoder=args.encoder,
        )
        child_model.model_x_to_c.load_state_dict(model_x_to_c_state_dict, strict=False)

        branch_state = {}
        branch_prefix = f"branches_c_to_y.{i}.model_c_to_y."
        for key, value in ensemble_state_dict.items():
            if key.startswith(branch_prefix):
                new_key = key.replace(branch_prefix, "")
                branch_state[new_key] = value
        if not branch_state:
            print(f"[Extract] Warning: no weights for C->Y branch {i} using prefix '{branch_prefix}'. Skipping.")
            continue

        child_model.model_c_to_y.load_state_dict(branch_state, strict=False)
        child_model.to(device).eval()
        child_models.append(child_model)

    if len(child_models) == 0:
        raise RuntimeError("No child models were extracted for DivEns.")
    return child_models


def load_independent_models(model_path, args, device, num_concepts, num_classes):
    """random-init baseline saved as list[state_dict]."""
    print(f"[Extract] random: loading list of independent models from {model_path}")
    list_of_state_dicts = torch.load(model_path, map_location="cpu")
    if not isinstance(list_of_state_dicts, list):
        raise TypeError(f"Expected a list of state_dicts from {model_path}, got {type(list_of_state_dicts)}")

    child_models = []
    for i, state_dict in enumerate(list_of_state_dicts):
        child_model = SingleE2EBranch(
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=num_classes,
            use_aux=getattr(args, "use_aux", False),
            n_attributes=num_concepts,
            expand_dim=args.expand_dim,
            encoder=args.encoder,
        )
        child_model.load_state_dict(state_dict, strict=False)
        child_model.to(device).eval()
        child_models.append(child_model)
    if len(child_models) == 0:
        raise RuntimeError("No independent models were loaded.")
    return child_models


def load_child_models_Partial(ensemble_or_ckpt, args, device=None, eval_mode=True, strict=False):
    """Extract per-branch child models from EnsembleWithPartialSharing."""

    class _XtoC(nn.Module):
        def __init__(self, trunk, deep_layers, concept_head, adapter=None):
            super().__init__()
            self.trunk = copy.deepcopy(trunk)
            self.deep_layers = copy.deepcopy(deep_layers)
            self.concept_head = copy.deepcopy(concept_head)
            self.adapter = copy.deepcopy(adapter) if adapter is not None else None

        def forward(self, x):
            feats = self.trunk(x)
            feats = self.deep_layers(feats)
            if self.adapter is not None:
                feats = self.adapter(feats)
            pooled = F.adaptive_avg_pool2d(feats, (1, 1))
            flat = torch.flatten(pooled, 1)
            concept_preds = [fc(flat) for fc in self.concept_head]
            return concept_preds

    class _ChildModel(nn.Module):
        def __init__(self, trunk, deep_layers, concept_head, classifier, adapter=None):
            super().__init__()
            self.model_x_to_c = _XtoC(trunk, deep_layers, concept_head, adapter)
            self.model_c_to_y = copy.deepcopy(classifier)

        def forward(self, x):
            concept_preds = self.model_x_to_c(x)
            concat = torch.cat(concept_preds, dim=1)
            y_pred = self.model_c_to_y(concat)
            return y_pred, concept_preds

    from collections import OrderedDict

    def _unwrap_state_dict(sd):
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], (dict, OrderedDict)):
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], (dict, OrderedDict)):
            sd = sd["model"]
        if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd

    def _infer_num_models_from_keys(sd, fallback=1):
        idxs = []
        for k in sd.keys():
            if isinstance(k, str) and k.startswith("branches."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    idxs.append(int(parts[1]))
        return (max(idxs) + 1) if idxs else fallback

    def _ensure_model(obj):
        if isinstance(obj, nn.Module):
            return obj
        if isinstance(obj, str):
            raw = torch.load(obj, map_location="cpu")
        elif isinstance(obj, (dict, OrderedDict)):
            raw = obj
        else:
            raise TypeError("ensemble_or_ckpt must be nn.Module / str / Dict")
        sd = _unwrap_state_dict(raw)
        inferred_num = _infer_num_models_from_keys(sd, fallback=getattr(args, "num_models", 1))
        model = EnsembleWithPartialSharing(
            num_models=inferred_num,
            num_classes=N_CLASSES,
            n_attributes=CONCEPT_DIM,
            encoder=getattr(args, "encoder", "resnet18"),
            expand_dim=getattr(args, "expand_dim", 0),
            split_point=getattr(args, "split_point", "layer4"),
            bottleneck_dim=getattr(args, "bottleneck_dim", 64),
        )
        model.load_state_dict(sd, strict=strict)
        return model

    ensemble = _ensure_model(ensemble_or_ckpt)
    children = []
    for b in ensemble.branches:
        adapter = getattr(b, "adapter", None) if hasattr(b, "adapter") else None
        child = _ChildModel(
            trunk=ensemble.trunk,
            deep_layers=b["deep_layers"],
            concept_head=b["concept_head"],
            classifier=b["classifier"],
            adapter=adapter,
        )
        if device is not None:
            child.to(device)
        if eval_mode:
            child.eval()
        children.append(child)
    return children


def load_child_models_Lora(model_path: str, args, device, num_concepts: int, num_classes: int) -> List[nn.Module]:
    """Extract per-adapter child models from a LoRA ensemble checkpoint.

    Mirrors the construction logic in `evaluation.py`:
      - share_mask all-zeros  => LoraEnsemble (no shared LoRA blocks)
      - share_mask all-ones   => LoraEnsembleshared (all shared)
      - otherwise             => LoraEnsemblePshared (partial sharing)
    """
    share_mask = str(getattr(args, "share_mask", ""))
    if set(share_mask) - set("01") or len(share_mask) != 12:
        raise ValueError(
            "For exp='Lora', --share_mask must be a 0/1 string of length 12 "
            "(ViT-small/base has 12 blocks), e.g. '000000000000' or '111111111111'."
        )

    lora_r = int(getattr(args, "lora_r", 8))
    lora_alpha = int(getattr(args, "lora_alpha", 16))
    lora_dropout = float(getattr(args, "lora_dropout", 0.1))

    num_models = int(getattr(args, "num_models", 0) or 0)
    if num_models <= 0:
        raise ValueError("For exp='Lora', --num_models must be > 0.")

    # Load checkpoint to CPU first; moving weights directly to CUDA can be much slower on some clusters
    # and increases GPU memory pressure during model reconstruction.
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, nn.Module):
        base = obj
    else:
        if share_mask == "0" * len(share_mask):
            base = LoraEnsemble(
                num_models=num_models,
                num_classes=num_classes,
                n_attributes=num_concepts,
                encoder=getattr(args, "encoder", "vit"),
                expand_dim=getattr(args, "expand_dim", 0),
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_block_mask=share_mask,
                backbone_pretrained=False,
            )
        elif share_mask == "1" * len(share_mask):
            base = LoraEnsembleshared(
                num_models=num_models,
                num_classes=num_classes,
                n_attributes=num_concepts,
                encoder=getattr(args, "encoder", "vit"),
                expand_dim=getattr(args, "expand_dim", 0),
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_block_mask=share_mask,
                backbone_pretrained=False,
            )
        else:
            base = LoraEnsemblePshared(
                num_models=num_models,
                num_classes=num_classes,
                n_attributes=num_concepts,
                encoder=getattr(args, "encoder", "vit"),
                expand_dim=getattr(args, "expand_dim", 0),
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_block_mask=share_mask,
                backbone_pretrained=False,
            )
        print(f"[Extract] Lora: loading from {model_path} ...")
        base.load_state_dict(obj, strict=True)

    base.to(device).eval()

    def _adapter_for(i: int):
        if isinstance(base, LoraEnsemblePshared):
            return [f"lora_{i}", "lora_shared"]
        if isinstance(base, LoraEnsemble):
            return f"lora_{i}"
        return "lora_shared"

    def _reset_adapter():
        if isinstance(base, LoraEnsemblePshared):
            base.lora_models.set_adapter(list(base.adapter_set))
        elif isinstance(base, LoraEnsemble):
            base.lora_models.set_adapter(str(base.adapter_set[0]))
        else:
            base.lora_models.set_adapter("lora_shared")

    class _LoraChild(nn.Module):
        def __init__(self, idx: int):
            super().__init__()
            self.idx = int(idx)
            self.lora_models = base.lora_models
            self.concept_head = base.final_heads[self.idx]["concept_head"]
            self.classifier = base.final_heads[self.idx]["classifier"]

        def model_x_to_c(self, x: torch.Tensor):
            self.lora_models.set_adapter(_adapter_for(self.idx))
            outputs = self.lora_models(x)
            feats = outputs.last_hidden_state[:, 0]
            concept_preds = [fc(feats) for fc in self.concept_head]
            _reset_adapter()
            return concept_preds

        def model_c_to_y(self, c_concat: torch.Tensor):
            return self.classifier(c_concat)

        def forward(self, x: torch.Tensor, is_training: bool = False):
            concept_preds = self.model_x_to_c(x)
            y_pred = self.model_c_to_y(torch.cat(concept_preds, dim=1))
            return y_pred, concept_preds

    return [_LoraChild(i).eval() for i in range(num_models)]


class _BackboneView(nn.Module):
    def __init__(self, stem, l1, l2, l3, l4):
        super().__init__()
        self.conv1 = stem["conv1"]
        self.bn1 = stem["bn1"]
        self.relu = stem["relu"]
        self.maxpool = stem["maxpool"]
        self.layer1, self.layer2, self.layer3, self.layer4 = l1, l2, l3, l4
        self.aux_logits = False

    def forward(self, x):
        raise NotImplementedError


def _collect_adapters_for_branch(ensemble, i, deepcopy_modules=True):
    ad = nn.ModuleDict()
    def _dc(m):
        return copy.deepcopy(m) if deepcopy_modules else m
    if hasattr(ensemble, "shared_adapters"):
        for k, v in ensemble.shared_adapters.items():
            ad[k] = _dc(v)
    if hasattr(ensemble, "branch_adapters"):
        branch = ensemble.branch_adapters[i]
        for k, v in branch.items():
            ad[k] = _dc(v)
    return ad


class _XtoCWithAdapters(nn.Module):
    def __init__(self, backbone: nn.Module, adapters: nn.ModuleDict, concept_head: nn.ModuleList, use_checkpoint=True):
        super().__init__()
        self.backbone = backbone
        self.adapters = adapters
        self.concept_head = concept_head
        self.use_checkpoint = use_checkpoint
        self.aux_logits = False

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                if getattr(m, "affine", False):
                    if m.weight is not None:
                        m.weight.requires_grad_(False)
                    if m.bias is not None:
                        m.bias.requires_grad_(False)

    @staticmethod
    def _k(name: str) -> str:
        return name.replace(".", "_")

    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        bb, ad = self.backbone, self.adapters
        out = bb.conv1(x) + ad[self._k("conv1")](x)
        out = bb.relu(bb.bn1(out))
        out = bb.maxpool(out)
        for li in range(1, 5):
            layer = getattr(bb, f"layer{li}")
            for bi, blk in enumerate(layer):
                def _blk(inp, _blk=blk, _li=li, _bi=bi):
                    identity = inp
                    if _blk.downsample is not None:
                        identity = _blk.downsample(identity)
                    res = _blk.conv1(inp) + ad[self._k(f"layer{_li}.{_bi}.conv1")](inp)
                    res = _blk.relu(_blk.bn1(res))
                    res = _blk.conv2(res) + ad[self._k(f"layer{_li}.{_bi}.conv2")](res)
                    res = _blk.bn2(res)
                    return _blk.relu(res + identity)

                out = _ckpt(_blk, out) if self.use_checkpoint else _blk(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor):
        self.backbone.eval()
        feats = self._forward_with_adapters(x)
        return [fc(feats) for fc in self.concept_head]


class _ChildForSafeguard(nn.Module):
    def __init__(self, x_to_c: nn.Module, c_to_y: nn.Module):
        super().__init__()
        self.model_x_to_c = x_to_c
        self.model_c_to_y = c_to_y

    def forward(self, x, is_training: bool = False):
        c_list = self.model_x_to_c(x)
        y = self.model_c_to_y(torch.cat(c_list, dim=1))
        return y, c_list


def _extract_from_sconvpar_ensemble(ensemble, device, deepcopy_modules=True, share_backbone=False, use_checkpoint=True):
    n = int(ensemble.num_models)
    children = []
    base_view = _BackboneView(ensemble.stem, ensemble.layer1, ensemble.layer2, ensemble.layer3, ensemble.layer4)
    if share_backbone:
        backbones = [base_view] * n
    else:
        backbones = [copy.deepcopy(base_view) if deepcopy_modules else base_view for _ in range(n)]
    for i in range(n):
        adapters = _collect_adapters_for_branch(ensemble, i, deepcopy_modules=deepcopy_modules)
        heads = ensemble.final_heads[i]
        concept = copy.deepcopy(heads["concept_head"]) if deepcopy_modules else heads["concept_head"]
        clf = copy.deepcopy(heads["classifier"]) if deepcopy_modules else heads["classifier"]
        x_to_c = _XtoCWithAdapters(backbones[i], adapters, concept, use_checkpoint=use_checkpoint)
        child = _ChildForSafeguard(x_to_c, clf).to(device).eval()
        children.append(child)
    return children


def load_child_models_flex(
    ensemble_model_path,
    args,
    device,
    num_concepts,
    num_classes,
    *,
    deepcopy_modules=True,
    share_backbone=False,
    use_checkpoint=True,
):
    """ConvAda extraction (flex)"""
    obj = torch.load(ensemble_model_path, map_location="cpu")
    if isinstance(obj, nn.Module):
        ensemble = obj
        return _extract_from_sconvpar_ensemble(
            ensemble,
            device,
            deepcopy_modules=deepcopy_modules,
            share_backbone=share_backbone,
            use_checkpoint=use_checkpoint,
        )

    if isinstance(obj, dict):
        # Infer num_models if missing
        num_models = getattr(args, "num_models", None)
        if num_models is None:
            idxs = []
            for k in obj.keys():
                if str(k).startswith("branch_adapters."):
                    try:
                        idxs.append(int(str(k).split(".")[1]))
                    except Exception:
                        pass
            num_models = max(idxs) + 1 if idxs else 1

        encoder = getattr(args, "encoder", "resnet18")
        expand_dim = getattr(args, "expand_dim", 0)
        bottleneck_dim = getattr(args, "bottleneck_dim", 64)
        share_mask = getattr(args, "share_mask", "11110")

        ensemble = SConvParEnsemble(
            num_models=num_models,
            num_classes=num_classes,
            n_attributes=num_concepts,
            encoder=encoder,
            bottleneck_dim=bottleneck_dim,
            expand_dim=expand_dim,
            share_mask=share_mask,
            use_pretrained=True,
        )
        ensemble.load_state_dict(obj, strict=False)
        return _extract_from_sconvpar_ensemble(
            ensemble,
            device,
            deepcopy_modules=deepcopy_modules,
            share_backbone=share_backbone,
            use_checkpoint=use_checkpoint,
        )

    raise TypeError("Cannot load model file")


def _load_child_models_for_exp(model_path: str, args, device, num_concepts: int, num_classes: int) -> List[nn.Module]:
    if args.exp == "X2C":
        return load_child_models_X2C(model_path, args, device, num_concepts, num_classes)
    if args.exp == "DivEns":
        return load_child_models_DivEns(model_path, args.num_models, device, args, num_concepts, num_classes)
    if args.exp == "random":
        return load_independent_models(model_path, args, device, num_concepts, num_classes)
    if args.exp == "Lora":
        return load_child_models_Lora(model_path, args, device, num_concepts, num_classes)
    if args.exp == "original":
        return [_load_single_model_for_original(model_path, args, device, num_concepts, num_classes)]
    if args.exp == "PartialX2C":
        return load_child_models_Partial(torch.load(model_path, map_location="cpu"), args=args, device=device, eval_mode=True)
    if args.exp == "ConvAda":
        if getattr(args, "encoder", "resnet18") != "resnet18":
            raise ValueError("--exp ConvAda requires --encoder resnet18.")
        return load_child_models_flex(model_path, args=args, device=device, num_concepts=num_concepts, num_classes=num_classes)
    if args.exp == "ConvParX2C":
        # ConvParEnsemble extraction was in your earlier script; keep compatibility by falling back to strict error
        raise NotImplementedError("ConvParX2C extraction is not included in v4 script; use your earlier evaluator for ConvPar.")
    raise ValueError(f"Unknown experiment type: {args.exp}")

def _load_single_model_for_original(model_path: str, args, device, num_concepts: int, num_classes: int) -> nn.Module:
    """Load a single model used by the original (single-model) safeguard baseline.

    Supports:
      - list[state_dict] checkpoints (random baseline) -> take index 0
      - state_dict for SingleE2EBranch
      - ensemble-style state_dicts -> extract member 0 (X2C/DivEns)
      - nn.Module objects -> use directly (or extract member 0 if it's an ensemble module)
    """
    obj = torch.load(model_path, map_location="cpu")

    if isinstance(obj, list):
        if len(obj) == 0:
            raise RuntimeError(f"Empty list checkpoint for original baseline: {model_path}")
        # List[state_dict] format (random baseline). For the original baseline we only need member-0,
        # so avoid reconstructing + loading *all* children again.
        state_dict0 = obj[0]
        if not isinstance(state_dict0, dict):
            raise TypeError(
                f"Expected list[dict] checkpoint for original baseline, got element type {type(state_dict0)}: {model_path}"
            )
        child_model = SingleE2EBranch(
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=num_classes,
            use_aux=getattr(args, "use_aux", False),
            n_attributes=num_concepts,
            expand_dim=args.expand_dim,
            encoder=args.encoder,
        )
        child_model.load_state_dict(state_dict0, strict=False)
        return child_model.to(device).eval()

    if isinstance(obj, nn.Module):
        # If it's an ensemble module we can extract children; otherwise use directly.
        if hasattr(obj, "num_models") and hasattr(obj, "final_heads"):
            children = _extract_from_sconvpar_ensemble(obj, device, deepcopy_modules=True, share_backbone=False, use_checkpoint=True)
            if not children:
                raise RuntimeError(f"Could not extract member 0 for original baseline from module checkpoint: {model_path}")
            return children[0]
        obj = obj.to(device).eval()
        return obj

    if isinstance(obj, dict):
        keys = list(obj.keys())
        if any(str(k).startswith("branches.0.") for k in keys):
            tmp = copy.deepcopy(args)
            tmp.exp = "X2C"
            tmp.num_models = 1
            children = load_child_models_X2C(model_path, tmp, device, num_concepts, num_classes)
            return children[0]
        if any(str(k).startswith("branches_c_to_y.0.") for k in keys) and any(str(k).startswith("model_x_to_c.") for k in keys):
            children = load_child_models_DivEns(model_path, 1, device, args, num_concepts, num_classes)
            return children[0]

        # Fall back to a plain SingleE2EBranch state_dict.
        model = SingleE2EBranch(
            n_class_attr=args.n_class_attr,
            pretrained=False,
            freeze=False,
            num_classes=num_classes,
            use_aux=getattr(args, "use_aux", False),
            n_attributes=num_concepts,
            expand_dim=args.expand_dim,
            encoder=args.encoder,
        )
        model.load_state_dict(obj, strict=False)
        return model.to(device).eval()

    raise TypeError(f"Unsupported checkpoint type for original baseline: {type(obj)} ({model_path})")

def _as_prob_binary(x: torch.Tensor) -> torch.Tensor:
    if x.min().item() >= 0.0 and x.max().item() <= 1.0:
        return x
    return torch.sigmoid(x)


def _as_prob_multiclass(y_logits: torch.Tensor) -> torch.Tensor:
    if (
        y_logits.min().item() >= 0.0
        and y_logits.max().item() <= 1.0
        and torch.allclose(y_logits.sum(dim=-1), torch.ones_like(y_logits.sum(dim=-1)), atol=1e-3)
    ):
        return y_logits
    return torch.softmax(y_logits, dim=-1)


def _binary_ce(pred: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
    if pred.min().item() >= 0.0 and pred.max().item() <= 1.0:
        return F.binary_cross_entropy(pred, target01, reduction="none")
    return F.binary_cross_entropy_with_logits(pred, target01, reduction="none")


def _concat_concepts(c_out) -> Optional[torch.Tensor]:
    if c_out is None:
        return None
    if torch.is_tensor(c_out):
        return c_out if c_out.dim() == 2 else c_out.view(c_out.size(0), -1)
    if isinstance(c_out, (list, tuple)):
        cols = []
        for t in c_out:
            if torch.is_tensor(t):
                cols.append(t.view(t.size(0), -1))
        return torch.cat(cols, dim=1) if cols else None
    return None


def _member_forward(model: nn.Module, x: torch.Tensor, *, num_classes: Optional[int] = None):
    try:
        out = model(x, is_training=False)
    except TypeError:
        out = model(x)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        a, b = out
        if isinstance(a, (list, tuple)) and torch.is_tensor(b):
            return b, a
        if isinstance(b, (list, tuple)) and torch.is_tensor(a):
            return a, b
        if torch.is_tensor(a) and torch.is_tensor(b) and num_classes is not None:
            if a.dim() == 2 and a.size(1) == num_classes:
                return a, b
            if b.dim() == 2 and b.size(1) == num_classes:
                return b, a
        if torch.is_tensor(a):
            return a, b

    if torch.is_tensor(out):
        return out, None
    raise TypeError(f"Unsupported forward output type: {type(out)}")


def _weighted_vote_consensus(y_probs_M: torch.Tensor, w_M: torch.Tensor) -> Tuple[float, int]:
    """For a single sample: y_probs_M is [M,C], w_M is [M]."""
    _, C = y_probs_M.shape
    y_hat_m = torch.argmax(y_probs_M, dim=1)  # [M]
    acc = torch.zeros(C, device=y_probs_M.device)
    acc.scatter_add_(0, y_hat_m, w_M)
    best_label = int(torch.argmax(acc).item())
    consensus = float(acc[best_label].item() / (w_M.sum().item() + 1e-12))
    return consensus, best_label


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp(min=eps)
    return float(-(p * torch.log(p)).sum().item())


def _update_weights_from_feedback(
    c_logits_MK: torch.Tensor,
    feedback: Dict[int, int],
    alpha: float,
    prior_logw_M: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M, K = c_logits_MK.shape
    device = c_logits_MK.device
    logw = torch.zeros(M, device=device) if prior_logw_M is None else prior_logw_M.clone()

    for k, v in feedback.items():
        if k < 0 or k >= K:
            continue
        target = torch.full((M,), float(v), device=device)
        pred_k = c_logits_MK[:, k]
        loss = _binary_ce(pred_k, target)
        logw = logw - alpha * loss

    return torch.softmax(logw, dim=0)


def _pick_next_concept_info_gain(
    y_probs_M: torch.Tensor,
    c_logits_MK: torch.Tensor,
    w_M: torch.Tensor,
    asked: set,
    alpha: float,
    score: str = "entropy",
) -> Tuple[int, float]:
    _, K = c_logits_MK.shape

    p_group = (w_M[:, None] * y_probs_M).sum(dim=0)
    if score == "entropy":
        s0 = _entropy(p_group)
    elif score == "1-consensus":
        cons, _ = _weighted_vote_consensus(y_probs_M, w_M)
        s0 = 1.0 - cons
    else:
        raise ValueError(f"Unknown score: {score}")

    best_k, best_gain = -1, 0.0
    c_prob_MK = _as_prob_binary(c_logits_MK)

    for k in range(K):
        if k in asked:
            continue
        pi = float((w_M * c_prob_MK[:, k]).sum().item())

        w1 = _update_weights_from_feedback(c_logits_MK, {k: 1}, alpha, prior_logw_M=torch.log(w_M + 1e-12))
        w0 = _update_weights_from_feedback(c_logits_MK, {k: 0}, alpha, prior_logw_M=torch.log(w_M + 1e-12))

        p1 = (w1[:, None] * y_probs_M).sum(dim=0)
        p0 = (w0[:, None] * y_probs_M).sum(dim=0)

        if score == "entropy":
            s1 = _entropy(p1)
            s0v = _entropy(p0)
        else:
            cons1, _ = _weighted_vote_consensus(y_probs_M, w1)
            cons0, _ = _weighted_vote_consensus(y_probs_M, w0)
            s1 = 1.0 - cons1
            s0v = 1.0 - cons0

        expected_s = pi * s1 + (1.0 - pi) * s0v
        gain = s0 - expected_s
        if gain > best_gain:
            best_gain = gain
            best_k = k
    return best_k, float(best_gain)


def rashomon_safeguard_predict_one(
    x: torch.Tensor,
    child_models: Sequence[nn.Module],
    *,
    num_classes: Optional[int] = None,
    expert_oracle: Optional[Callable[[int], int]] = None,
    init_feedback: Optional[Dict[int, int]] = None,
    budget: int = 8,
    consensus_thresh: float = 0.80,
    alpha: float = 5.0,
    score: str = "entropy",
) -> Dict[str, object]:
    assert x.dim() >= 2 and x.size(0) == 1
    device = x.device
    M = len(child_models)
    if M == 0:
        return {"accept": False, "reason": "no_members"}

    y_logits_list = []
    c_logits_list = []
    with torch.no_grad():
        for m in child_models:
            y_m, c_m = _member_forward(m, x, num_classes=num_classes)
            y_logits_list.append(y_m)
            c_logits_list.append(_concat_concepts(c_m))

    y_logits_M = torch.cat(y_logits_list, dim=0)
    y_probs_M = _as_prob_multiclass(y_logits_M)

    if any(c is None for c in c_logits_list):
        w = torch.full((M,), 1.0 / M, device=device)
        cons, yhat = _weighted_vote_consensus(y_probs_M, w)
        return {"accept": True, "yhat": yhat, "consensus": cons, "queries": [], "weights": w.detach().cpu(), "note": "no_concepts"}

    c_logits_MK = torch.cat(c_logits_list, dim=0)
    feedback = dict(init_feedback) if init_feedback else {}
    asked = set(feedback.keys())
    queries: List[Tuple[int, int, float]] = []

    w = _update_weights_from_feedback(c_logits_MK, feedback, alpha)
    cons, yhat = _weighted_vote_consensus(y_probs_M, w)
    if cons >= consensus_thresh:
        return {"accept": True, "yhat": yhat, "consensus": cons, "queries": queries, "weights": w.detach().cpu()}

    for _ in range(budget):
        k, gain = _pick_next_concept_info_gain(y_probs_M, c_logits_MK, w, asked, alpha, score=score)
        if k < 0:
            break
        asked.add(k)
        if expert_oracle is None:
            return {"accept": False, "reason": "no_oracle", "yhat": yhat, "consensus": cons, "queries": queries, "weights": w.detach().cpu()}
        v = int(expert_oracle(k))
        queries.append((k, v, gain))
        w = _update_weights_from_feedback(c_logits_MK, {k: v}, alpha, prior_logw_M=torch.log(w + 1e-12))
        cons, yhat = _weighted_vote_consensus(y_probs_M, w)
        if cons >= consensus_thresh:
            return {"accept": True, "yhat": yhat, "consensus": cons, "queries": queries, "weights": w.detach().cpu()}

    return {"accept": False, "reason": "unresolved", "yhat": yhat, "consensus": cons, "queries": queries, "weights": w.detach().cpu()}

def _original_task_prob_mc(
    x: torch.Tensor,
    model: nn.Module,
    *,
    num_classes: int,
    fixed_concepts: Optional[Dict[int, int]] = None,
    mc_samples: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.size(0) == 1
    if not (hasattr(model, "model_x_to_c") and hasattr(model, "model_c_to_y")):
        raise AttributeError("Model must expose model_x_to_c and model_c_to_y")

    with torch.no_grad():
        c_list = model.model_x_to_c(x)
        c_logits = _concat_concepts(c_list)
        if c_logits is None:
            raise RuntimeError("Original safeguard requires concept outputs")
        q = _as_prob_binary(c_logits).view(-1)
        K = q.numel()
        fixed_concepts = fixed_concepts or {}
        q_eff = q.clone()
        for k, v in fixed_concepts.items():
            if 0 <= k < K:
                q_eff[k] = float(v)

        S = int(mc_samples)
        q_samp = q_eff.clamp(1e-6, 1 - 1e-6)
        c_samples = torch.bernoulli(q_samp.expand(S, K))
        for k, v in fixed_concepts.items():
            if 0 <= k < K:
                c_samples[:, k] = float(v)

        y_logits = model.model_c_to_y(c_samples)
        y_probs = torch.softmax(y_logits, dim=-1)
        p_y = y_probs.mean(dim=0)
    return p_y, q_eff




def _original_task_prob_proto(
    x: torch.Tensor,
    model: nn.Module,
    *,
    num_classes: int,
    fixed_concepts: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.size(0) == 1, "single sample only"
    x2c = _get_x2c_module(model)
    c2y = _get_c2y_module(model)

    with torch.no_grad():
        c_list = x2c(x)
        c_logits = _concat_concepts(c_list)  # [1,K]
        if c_logits is None:
            raise RuntimeError("Original safeguard requires concept outputs.")

        q = _as_prob_binary(c_logits).view(-1)  # [K]
        K = q.numel()
        fixed_concepts = fixed_concepts or {}

        q_eff = q.clone()
        for k, v in fixed_concepts.items():
            if 0 <= k < K:
                q_eff[k] = float(v)

        y_logits = c2y(q_eff.view(1, -1))  # [1,C]
        p_y = torch.softmax(y_logits.view(-1), dim=-1)  # [C]

    return p_y, q_eff


def _pick_next_concept_original_gain_proto(
    x: torch.Tensor,
    model: nn.Module,
    *,
    num_classes: int,
    asked: set,
    fixed_concepts: Dict[int, int],
    score: str,
) -> Tuple[int, float]:
    """Choose next concept by expected uncertainty reduction using the prototype approximation."""
    p_base, q_eff = _original_task_prob_proto(
        x, model, num_classes=num_classes, fixed_concepts=fixed_concepts
    )
    if score == "entropy":
        s_base = _entropy(p_base)
    else:
        s_base = 1.0 - float(p_base.max().item())

    K = q_eff.numel()
    best_k, best_gain = -1, 0.0

    for k in range(K):
        if k in asked:
            continue

        pi = float(q_eff[k].item())

        p1, _ = _original_task_prob_proto(
            x, model, num_classes=num_classes, fixed_concepts={**fixed_concepts, k: 1}
        )
        p0, _ = _original_task_prob_proto(
            x, model, num_classes=num_classes, fixed_concepts={**fixed_concepts, k: 0}
        )

        if score == "entropy":
            s1 = _entropy(p1)
            s0 = _entropy(p0)
        else:
            s1 = 1.0 - float(p1.max().item())
            s0 = 1.0 - float(p0.max().item())

        expected_s = pi * s1 + (1.0 - pi) * s0
        gain = s_base - expected_s

        if gain > best_gain:
            best_gain = gain
            best_k = k

    return best_k, float(best_gain)

def _pick_next_concept_original_gain(
    x: torch.Tensor,
    model: nn.Module,
    *,
    num_classes: int,
    asked: set,
    fixed_concepts: Dict[int, int],
    mc_samples: int,
    score: str,
) -> Tuple[int, float]:
    p_base, q_eff = _original_task_prob_mc(x, model, num_classes=num_classes, fixed_concepts=fixed_concepts, mc_samples=mc_samples)
    if score == "entropy":
        s_base = _entropy(p_base)
    else:
        s_base = 1.0 - float(p_base.max().item())
    K = q_eff.numel()
    best_k, best_gain = -1, 0.0
    for k in range(K):
        if k in asked:
            continue
        pi = float(q_eff[k].item())
        p1, _ = _original_task_prob_mc(x, model, num_classes=num_classes, fixed_concepts={**fixed_concepts, k: 1}, mc_samples=mc_samples)
        p0, _ = _original_task_prob_mc(x, model, num_classes=num_classes, fixed_concepts={**fixed_concepts, k: 0}, mc_samples=mc_samples)
        if score == "entropy":
            s1 = _entropy(p1)
            s0 = _entropy(p0)
        else:
            s1 = 1.0 - float(p1.max().item())
            s0 = 1.0 - float(p0.max().item())
        expected_s = pi * s1 + (1.0 - pi) * s0
        gain = s_base - expected_s
        if gain > best_gain:
            best_gain = gain
            best_k = k
    return best_k, float(best_gain)


def original_safeguard_predict_one(
    x: torch.Tensor,
    model: nn.Module,
    *,
    num_classes: int,
    expert_oracle: Optional[Callable[[int], int]] = None,
    init_feedback: Optional[Dict[int, int]] = None,
    budget: int = 8,
    conf_thresh: float = 0.80,
    mc_samples: int = 64,
    score: str = "entropy",
) -> Dict[str, object]:
    fixed: Dict[int, int] = dict(init_feedback) if init_feedback else {}
    asked = set(fixed.keys())
    queries: List[Tuple[int, int, float]] = []

    p_y, _ = _original_task_prob_mc(x, model, num_classes=num_classes, fixed_concepts=fixed, mc_samples=mc_samples)
    yhat = int(torch.argmax(p_y).item())
    conf = float(p_y.max().item())
    if conf >= conf_thresh:
        return {"accept": True, "yhat": yhat, "conf": conf, "queries": queries}

    for _ in range(budget):
        k, gain = _pick_next_concept_original_gain(
            x,
            model,
            num_classes=num_classes,
            asked=asked,
            fixed_concepts=fixed,
            mc_samples=mc_samples,
            score=score,
        )
        if k < 0:
            break
        asked.add(k)
        if expert_oracle is None:
            return {"accept": False, "reason": "no_oracle", "yhat": yhat, "conf": conf, "queries": queries}
        v = int(expert_oracle(k))
        fixed[k] = v
        queries.append((k, v, gain))
        p_y, _ = _original_task_prob_mc(x, model, num_classes=num_classes, fixed_concepts=fixed, mc_samples=mc_samples)
        yhat = int(torch.argmax(p_y).item())
        conf = float(p_y.max().item())
        if conf >= conf_thresh:
            return {"accept": True, "yhat": yhat, "conf": conf, "queries": queries}

    return {"accept": False, "reason": "unresolved", "yhat": yhat, "conf": conf, "queries": queries}

def _compute_uncertain_flags(
    *,
    test_loader,
    child_models: Sequence[nn.Module],
    class_idx_set: Optional[set[int]],
    device: torch.device,
    sg_consensus: float,
    dataname: str,
    args,
) -> Tuple[List[bool], Dict[str, int]]:
    """Return uncertain_flags aligned with iteration order over class-filtered samples.

    uncertain(x) := consensus0(x) < sg_consensus under UNIFORM weights.
    """
    M = len(child_models)
    w = torch.full((M,), 1.0 / M, device=device)

    flags: List[bool] = []
    n_selected = 0
    n_uncertain = 0
    skipped_by_class = 0

    with torch.no_grad():
        for data in test_loader:
            if dataname == "CUB":
                inputs, y_true, _c_true = data
                x_batch = inputs.to(device) if torch.is_tensor(inputs) else inputs
            else:
                x_batch = data["img"].to(device)
                y_true = data["class_label"]

            B = x_batch.size(0)
            for b in range(B):
                y = int(y_true[b].item()) if torch.is_tensor(y_true) else int(y_true[b])
                if class_idx_set is not None and y not in class_idx_set:
                    skipped_by_class += 1
                    continue
                x = x_batch[b : b + 1]
                y_logits_list = []
                for m in child_models:
                    y_m, _c_m = _member_forward(m, x, num_classes=N_CLASSES)
                    y_logits_list.append(y_m)
                y_logits_M = torch.cat(y_logits_list, dim=0)
                y_probs_M = _as_prob_multiclass(y_logits_M)
                cons, _ = _weighted_vote_consensus(y_probs_M, w)
                is_unc = cons < sg_consensus
                flags.append(is_unc)
                n_selected += 1
                if is_unc:
                    n_uncertain += 1

    return flags, {
        "n_selected_by_class": n_selected,
        "n_uncertain0_in_selected": n_uncertain,
        "skipped_by_class_filter": skipped_by_class,
    }

def _parse_csv_floats(s: str) -> List[float]:
    s = str(s or "").strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    out: List[float] = []
    for p in parts:
        if not p:
            continue
        out.append(float(p))
    return out


def _parse_csv_ints(s: str) -> List[int]:
    s = str(s or "").strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def _uniform_consensus_batch(y_probs_MBC: torch.Tensor) -> torch.Tensor:
    """Compute consensus0 under uniform weights.

    Args:
      y_probs_MBC: [M,B,C] probabilities.
    Returns:
      cons_B: [B] float tensor on CPU.
    """
    M, B, C = y_probs_MBC.shape
    y_hat_MB = torch.argmax(y_probs_MBC, dim=-1).detach().cpu().numpy()  # [M,B]
    cons = np.zeros((B,), dtype=np.float32)
    for b in range(B):
        counts = np.bincount(y_hat_MB[:, b], minlength=C)
        cons[b] = float(counts.max() / max(1, M))
    return torch.from_numpy(cons)


def _member_trajectory_from_logits(
    *,
    y_probs_M: torch.Tensor,        # [M,C]
    c_logits_MK: Optional[torch.Tensor],  # [M,K] or None
    c_oracle01: Optional[torch.Tensor],   # [K] or None
    budget_max: int,
    alpha: float,
    score: str,
    stop_tau: Optional[float] = None,
) -> Dict[str, object]:
    """Precompute the full query trajectory up to budget_max (independent of tau).

    Returns:
      - cons_seq: List[float], length = steps+1 (includes t=0)
      - yhat_seq: List[int], aligned with cons_seq
      - queries:  List[Tuple[k,v,gain]], length = steps (<= budget_max)
    """
    assert y_probs_M.dim() == 2
    M, _C = y_probs_M.shape
    w = torch.full((M,), 1.0 / max(1, M), device=y_probs_M.device, dtype=y_probs_M.dtype)
    cons0, yhat0 = _weighted_vote_consensus(y_probs_M, w)

    cons_seq: List[float] = [float(cons0)]
    yhat_seq: List[int] = [int(yhat0)]
    queries: List[Tuple[int, int, float]] = []

    if stop_tau is not None and float(cons0) >= float(stop_tau):
        return {"cons_seq": cons_seq, "yhat_seq": yhat_seq, "queries": queries}

    if budget_max <= 0 or c_logits_MK is None or c_oracle01 is None:
        return {"cons_seq": cons_seq, "yhat_seq": yhat_seq, "queries": queries}

    c_logits_MK = c_logits_MK.to(device=y_probs_M.device, dtype=y_probs_M.dtype)
    c_oracle01 = c_oracle01.to(device=y_probs_M.device)

    asked: set = set()
    for _ in range(int(budget_max)):
        k, gain = _pick_next_concept_info_gain(
            y_probs_M=y_probs_M,
            c_logits_MK=c_logits_MK,
            w_M=w,
            asked=asked,
            alpha=float(alpha),
            score=str(score),
        )
        if k < 0:
            break
        asked.add(k)
        v = int(c_oracle01[k].item())
        queries.append((int(k), int(v), float(gain)))

        w = _update_weights_from_feedback(
            c_logits_MK,
            {int(k): int(v)},
            float(alpha),
            prior_logw_M=torch.log(w + 1e-12),
        )
        cons_t, yhat_t = _weighted_vote_consensus(y_probs_M, w)
        cons_seq.append(float(cons_t))
        yhat_seq.append(int(yhat_t))
        if stop_tau is not None and float(cons_t) >= float(stop_tau):
            break

    return {"cons_seq": cons_seq, "yhat_seq": yhat_seq, "queries": queries}


def _original_task_prob_mc_from_logits(
    *,
    c_logits_1K: torch.Tensor,
    c2y: Callable[[torch.Tensor], torch.Tensor],
    num_classes: int,
    fixed_concepts: Optional[Dict[int, int]] = None,
    mc_samples: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC task prob from precomputed concept logits (avoids re-running X->C repeatedly)."""
    c_logits = c_logits_1K.view(-1)
    q = _as_prob_binary(c_logits).view(-1)
    K = int(q.numel())
    fixed_concepts = fixed_concepts or {}

    q_eff = q.clone()
    for k, v in fixed_concepts.items():
        if 0 <= int(k) < K:
            q_eff[int(k)] = float(v)

    S = int(mc_samples)
    q_samp = q_eff.clamp(1e-6, 1 - 1e-6)
    c_samples = torch.bernoulli(q_samp.expand(S, K))
    for k, v in fixed_concepts.items():
        if 0 <= int(k) < K:
            c_samples[:, int(k)] = float(v)

    y_logits = c2y(c_samples)  # [S,C]
    y_probs = torch.softmax(y_logits, dim=-1)
    p_y = y_probs.mean(dim=0)  # [C]
    return p_y, q_eff


def _pick_next_concept_original_gain_from_logits(
    *,
    c_logits_1K: torch.Tensor,
    c2y: Callable[[torch.Tensor], torch.Tensor],
    num_classes: int,
    asked: set,
    fixed_concepts: Dict[int, int],
    mc_samples: int,
    score: str,
) -> Tuple[int, float]:
    p_base, q_eff = _original_task_prob_mc_from_logits(
        c_logits_1K=c_logits_1K,
        c2y=c2y,
        num_classes=num_classes,
        fixed_concepts=fixed_concepts,
        mc_samples=mc_samples,
    )
    if score == "entropy":
        s_base = _entropy(p_base)
    else:
        s_base = 1.0 - float(p_base.max().item())

    K = int(q_eff.numel())
    best_k, best_gain = -1, 0.0
    for k in range(K):
        if k in asked:
            continue
        pi = float(q_eff[k].item())
        p1, _ = _original_task_prob_mc_from_logits(
            c_logits_1K=c_logits_1K,
            c2y=c2y,
            num_classes=num_classes,
            fixed_concepts={**fixed_concepts, k: 1},
            mc_samples=mc_samples,
        )
        p0, _ = _original_task_prob_mc_from_logits(
            c_logits_1K=c_logits_1K,
            c2y=c2y,
            num_classes=num_classes,
            fixed_concepts={**fixed_concepts, k: 0},
            mc_samples=mc_samples,
        )
        if score == "entropy":
            s1 = _entropy(p1)
            s0 = _entropy(p0)
        else:
            s1 = 1.0 - float(p1.max().item())
            s0 = 1.0 - float(p0.max().item())
        expected_s = pi * s1 + (1.0 - pi) * s0
        gain = s_base - expected_s
        if gain > best_gain:
            best_gain = gain
            best_k = k
    return int(best_k), float(best_gain)


def _original_trajectory_mc_from_logits(
    *,
    c_logits_1K: torch.Tensor,
    c2y: Callable[[torch.Tensor], torch.Tensor],
    c_oracle01: Optional[torch.Tensor],
    num_classes: int,
    budget_max: int,
    mc_samples: int,
    score: str,
    stop_tau: Optional[float] = None,
) -> Dict[str, object]:
    """Precompute conf/yhat trajectory up to budget_max for the original MC baseline."""
    fixed: Dict[int, int] = {}
    asked: set = set()
    queries: List[Tuple[int, int, float]] = []

    p_y, _q = _original_task_prob_mc_from_logits(
        c_logits_1K=c_logits_1K,
        c2y=c2y,
        num_classes=num_classes,
        fixed_concepts=fixed,
        mc_samples=mc_samples,
    )
    yhat = int(torch.argmax(p_y).item())
    conf = float(p_y.max().item())
    conf_seq: List[float] = [conf]
    yhat_seq: List[int] = [yhat]

    if stop_tau is not None and float(conf) >= float(stop_tau):
        return {"conf_seq": conf_seq, "yhat_seq": yhat_seq, "queries": queries}

    if budget_max <= 0 or c_oracle01 is None:
        return {"conf_seq": conf_seq, "yhat_seq": yhat_seq, "queries": queries}

    K = int(c_logits_1K.numel())
    for _ in range(int(budget_max)):
        k, gain = _pick_next_concept_original_gain_from_logits(
            c_logits_1K=c_logits_1K,
            c2y=c2y,
            num_classes=num_classes,
            asked=asked,
            fixed_concepts=fixed,
            mc_samples=int(mc_samples),
            score=str(score),
        )
        if k < 0 or k >= K:
            break
        asked.add(k)
        v = int(c_oracle01[k].item())
        fixed[k] = v
        queries.append((int(k), int(v), float(gain)))
        p_y, _q = _original_task_prob_mc_from_logits(
            c_logits_1K=c_logits_1K,
            c2y=c2y,
            num_classes=num_classes,
            fixed_concepts=fixed,
            mc_samples=mc_samples,
        )
        yhat = int(torch.argmax(p_y).item())
        conf = float(p_y.max().item())
        conf_seq.append(conf)
        yhat_seq.append(yhat)
        if stop_tau is not None and float(conf) >= float(stop_tau):
            break

    return {"conf_seq": conf_seq, "yhat_seq": yhat_seq, "queries": queries}


def _eval_member_from_cached_trajectory(
    *,
    run_tag: str,
    args,
    out_dir: str,
    summary_csv_path: str,
    model_path: str,
    y_true_N: torch.Tensor,              # [N]
    cons0_N: torch.Tensor,               # [N]
    traj_cons_seq: List[List[float]],    # len N, each len <= Bmax+1
    traj_yhat_seq: List[List[int]],      # len N
    traj_queries: List[List[Tuple[int, int, float]]],  # len N
    keep_flags: List[bool],
    sel_stats: Dict[str, int],
    uncertain_policy: str,
) -> None:
    class_idx_set = _parse_class_idxs(getattr(args, "class_idxs", "-1"))
    run_suffix = _make_run_suffix(run_tag, args, class_idx_set)
    overlap = getattr(args, "overlap", "all")

    n_selected = int(sel_stats["n_selected_by_class"])
    n_uncertain0 = int(sel_stats["n_uncertain0_in_selected"])
    skipped_by_class = int(sel_stats["skipped_by_class_filter"])
    n_uncertain_used = int(sum(bool(x) for x in keep_flags)) if overlap == "uncertain" else n_uncertain0

    total_eval = 0
    accepted = 0
    correct = 0
    total_queries = 0
    concept_query_count = np.zeros(CONCEPT_DIM, dtype=np.int64)
    concept_gain_sum = np.zeros(CONCEPT_DIM, dtype=np.float64)

    tau = float(getattr(args, "sg_consensus", 0.80))
    budget = int(getattr(args, "sg_budget", 8))

    for i, keep in enumerate(keep_flags):
        if not keep:
            continue
        total_eval += 1
        cons_seq = traj_cons_seq[i]
        yhat_seq = traj_yhat_seq[i]
        queries = traj_queries[i]

        # Find first acceptance time t <= budget.
        t_accept = None
        t_max = min(int(budget), len(cons_seq) - 1)
        for t in range(t_max + 1):
            if float(cons_seq[t]) >= tau:
                t_accept = t
                break

        if t_accept is not None:
            accepted += 1
            if int(yhat_seq[t_accept]) == int(y_true_N[i].item()):
                correct += 1
            q_used = int(t_accept)
        else:
            # Match original code: if not accepted, we spent the full budget (or whatever trajectory length allows).
            q_used = min(int(budget), len(queries))

        total_queries += q_used
        for (kk, _vv, gain) in queries[:q_used]:
            if 0 <= int(kk) < CONCEPT_DIM:
                concept_query_count[int(kk)] += 1
                concept_gain_sum[int(kk)] += float(gain)

    coverage = accepted / (total_eval + 1e-12)
    sel_acc = correct / (accepted + 1e-12)
    avg_q = total_queries / (total_eval + 1e-12)

    rashomon_txt = os.path.join(out_dir, f"rashomon_safeguard__{run_suffix}.txt")
    txt_lines = []
    txt_lines.append(f"Run tag: {run_tag}")
    txt_lines.append("method=member_reweighting")
    txt_lines.append(f"exp={args.exp}")
    txt_lines.append(f"ckpt={model_path}")
    txt_lines.append(f"dataname={args.dataname}")
    txt_lines.append(f"class_idxs={getattr(args,'class_idxs','-1')}")
    txt_lines.append(f"overlap={overlap}")
    txt_lines.append(f"uncertain_policy={uncertain_policy}")
    txt_lines.append(f"n_selected_by_class={n_selected}")
    txt_lines.append(f"n_uncertain0_in_selected={n_uncertain0}")
    txt_lines.append(f"skipped_by_class_filter={skipped_by_class}")
    txt_lines.append("")
    txt_lines.append("[Member safeguard]")
    txt_lines.append(f"coverage={coverage:.6f}")
    txt_lines.append(f"selective_accuracy={sel_acc:.6f}")
    txt_lines.append(f"avg_queries={avg_q:.6f}")
    txt_lines.append(f"total_evaluated={total_eval}")
    txt_lines.append(f"accepted={accepted}")
    txt_lines.append(f"correct={correct}")
    txt_lines.append("\nTop concepts by query frequency (k, count, avg_gain_when_queried):")
    top = np.argsort(-concept_query_count)[: min(50, CONCEPT_DIM)]
    for kk in top:
        if concept_query_count[kk] == 0:
            continue
        avg_gain = concept_gain_sum[kk] / max(1, concept_query_count[kk])
        txt_lines.append(f"{kk}\t{concept_query_count[kk]}\t{avg_gain:.6f}")

    _ensure_dir(out_dir)
    with open(rashomon_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    _append_summary_csv(
        summary_csv_path,
        {
            "tag": _sanitize_tag(run_tag),
            "method": "member_reweighting",
            "exp": args.exp,
            "ckpt": model_path,
            "dataname": args.dataname,
            "class_idxs": getattr(args, "class_idxs", "-1"),
            "overlap": overlap,
            "uncertain_policy": uncertain_policy,
            "n_uncertain_used": int(n_uncertain_used),
            "consensus_thresh": float(getattr(args, "sg_consensus", 0.80)),
            "budget": int(getattr(args, "sg_budget", 8)),
            "alpha": float(getattr(args, "sg_alpha", 5.0)),
            "score": str(getattr(args, "sg_score", "entropy")),
            "coverage": float(coverage),
            "selective_accuracy": float(sel_acc),
            "avg_queries": float(avg_q),
            "n_selected_by_class": int(n_selected),
            "n_uncertain0_in_selected": int(n_uncertain0),
            "total_evaluated": int(total_eval),
            "accepted": int(accepted),
            "correct": int(correct),
            "skipped_by_class_filter": int(skipped_by_class),
        },
    )


def _cache_original_run_trajectory_for_sweep(
    *,
    run_tag: str,
    args,
    test_loader,
    device: torch.device,
    budget_max: int,
    stop_tau: Optional[float] = None,
    progress_every: int = 250,
) -> Dict[str, object]:
    """Load original (single-model) baseline once and cache per-sample trajectories up to budget_max."""
    class_idx_set = _parse_class_idxs(getattr(args, "class_idxs", "-1"))

    model_path = getattr(args, "ckpt", "")
    if not model_path:
        log_dir = getattr(args, "log_dir", "")
        model_path = os.path.join(log_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found for run '{run_tag}': {model_path}")

    ref_model = _load_single_model_for_original(model_path, args, device, CONCEPT_DIM, N_CLASSES).to(device).eval()
    # The original baseline's inner loop is many small MLP calls; this is typically faster on CPU
    # (avoids GPU kernel-launch overhead), while keeping X->C on GPU.
    c2y_cpu = copy.deepcopy(ref_model.model_c_to_y).cpu().eval()

    y_true_list: List[int] = []
    conf0_list: List[float] = []
    traj_conf_seq: List[List[float]] = []
    traj_yhat_seq: List[List[int]] = []
    traj_queries: List[List[Tuple[int, int, float]]] = []

    n_selected = 0
    skipped_by_class = 0

    mc_samples = int(getattr(args, "orig_mc_samples", 64))
    score = str(getattr(args, "sg_score", "entropy"))

    t0 = time.time()
    last_print = t0

    with torch.no_grad():
        for data in test_loader:
            if str(getattr(args, "dataname", "CUB")) == "CUB":
                inputs, y_true, c_true = data
                x_batch = inputs.to(device) if torch.is_tensor(inputs) else inputs
            else:
                x_batch = data["img"].to(device)
                y_true = data["class_label"]
                c_true = data.get("attribute_label", None)

            B = x_batch.size(0)

            # X->C once per batch
            c_list = ref_model.model_x_to_c(x_batch)
            c_logits_BK = torch.cat([t.view(B, -1) for t in c_list], dim=1).detach().cpu()  # [B,K] on CPU

            for b in range(B):
                yb = int(y_true[b].item()) if torch.is_tensor(y_true) else int(y_true[b])
                if class_idx_set is not None and yb not in class_idx_set:
                    skipped_by_class += 1
                    continue

                if c_true is None:
                    c01 = None
                else:
                    c01 = c_true[b]
                    if torch.is_tensor(c01):
                        c01 = c01.float()
                        if c01.min().item() < 0:
                            c01 = (c01 > 0).float()
                        else:
                            c01 = (c01 > 0.5).float()
                        c01 = c01.view(-1).cpu()
                    else:
                        c01 = torch.tensor(c01).float().view(-1)
                        c01 = (c01 > 0.5).float()

                traj = _original_trajectory_mc_from_logits(
                    c_logits_1K=c_logits_BK[b],
                    c2y=c2y_cpu,
                    c_oracle01=c01,
                    num_classes=N_CLASSES,
                    budget_max=int(budget_max),
                    mc_samples=mc_samples,
                    score=score,
                    stop_tau=stop_tau,
                )
                conf_seq = list(traj["conf_seq"])
                yhat_seq = list(traj["yhat_seq"])
                queries = list(traj["queries"])

                y_true_list.append(int(yb))
                conf0_list.append(float(conf_seq[0]) if conf_seq else 0.0)
                traj_conf_seq.append(conf_seq)
                traj_yhat_seq.append(yhat_seq)
                traj_queries.append(queries)
                n_selected += 1

                if progress_every > 0 and (n_selected % int(progress_every) == 0):
                    now = time.time()
                    if now - last_print >= 1.0:
                        rate = n_selected / max(1e-9, now - t0)
                        print(f"[SweepCache][{run_tag}][Original] cached={n_selected} ({rate:.2f} samp/s)")
                        last_print = now

    # cleanup
    del c2y_cpu
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_true_N = torch.tensor(y_true_list, dtype=torch.long)
    conf0_N = torch.tensor(conf0_list, dtype=torch.float32)

    return {
        "run_tag": run_tag,
        "model_path": model_path,
        "y_true_N": y_true_N,
        "conf0_N": conf0_N,
        "traj_conf_seq": traj_conf_seq,
        "traj_yhat_seq": traj_yhat_seq,
        "traj_queries": traj_queries,
        "sel_stats_base": {
            "n_selected_by_class": int(n_selected),
            "skipped_by_class_filter": int(skipped_by_class),
        },
    }


def _eval_original_from_cached_trajectory(
    *,
    run_tag: str,
    args,
    out_dir: str,
    summary_csv_path: str,
    model_path: str,
    y_true_N: torch.Tensor,              # [N]
    conf0_N: torch.Tensor,               # [N]
    traj_conf_seq: List[List[float]],    # len N
    traj_yhat_seq: List[List[int]],      # len N
    traj_queries: List[List[Tuple[int, int, float]]],  # len N
    keep_flags: List[bool],
    sel_stats: Dict[str, int],
    uncertain_policy: str,
) -> None:
    class_idx_set = _parse_class_idxs(getattr(args, "class_idxs", "-1"))
    run_suffix = _make_run_suffix(run_tag, args, class_idx_set)
    overlap = getattr(args, "overlap", "all")

    n_selected = int(sel_stats["n_selected_by_class"])
    n_uncertain0 = int(sel_stats["n_uncertain0_in_selected"])
    skipped_by_class = int(sel_stats["skipped_by_class_filter"])
    n_uncertain_used = int(sum(bool(x) for x in keep_flags)) if overlap == "uncertain" else n_uncertain0

    total_eval = 0
    accepted = 0
    correct = 0
    total_queries = 0
    concept_query_count = np.zeros(CONCEPT_DIM, dtype=np.int64)
    concept_gain_sum = np.zeros(CONCEPT_DIM, dtype=np.float64)

    tau = float(getattr(args, "sg_consensus", 0.80))
    budget = int(getattr(args, "sg_budget", 8))

    for i, keep in enumerate(keep_flags):
        if not keep:
            continue
        total_eval += 1
        conf_seq = traj_conf_seq[i]
        yhat_seq = traj_yhat_seq[i]
        queries = traj_queries[i]

        t_accept = None
        t_max = min(int(budget), len(conf_seq) - 1)
        for t in range(t_max + 1):
            if float(conf_seq[t]) >= tau:
                t_accept = t
                break

        if t_accept is not None:
            accepted += 1
            if int(yhat_seq[t_accept]) == int(y_true_N[i].item()):
                correct += 1
            q_used = int(t_accept)
        else:
            q_used = min(int(budget), len(queries))

        total_queries += q_used
        for (kk, _vv, gain) in queries[:q_used]:
            if 0 <= int(kk) < CONCEPT_DIM:
                concept_query_count[int(kk)] += 1
                concept_gain_sum[int(kk)] += float(gain)

    coverage = accepted / (total_eval + 1e-12)
    sel_acc = correct / (accepted + 1e-12)
    avg_q = total_queries / (total_eval + 1e-12)

    original_txt = os.path.join(out_dir, f"original_safeguard__{run_suffix}.txt")
    txt2 = []
    txt2.append(f"Run tag: {run_tag}")
    txt2.append("baseline=original_concept_sampling")
    txt2.append(f"exp=original")
    txt2.append(f"ckpt={model_path}")
    txt2.append(f"dataname={args.dataname}")
    txt2.append(f"class_idxs={getattr(args,'class_idxs','-1')}")
    txt2.append(f"overlap={overlap}")
    txt2.append(f"uncertain_policy={uncertain_policy}")
    txt2.append(f"n_selected_by_class={n_selected}")
    txt2.append(f"n_uncertain0_in_selected={n_uncertain0}")
    txt2.append(f"skipped_by_class_filter={skipped_by_class}")
    txt2.append("")
    txt2.append(f"conf_thresh={tau:.6f}")
    txt2.append(f"budget={int(budget)}")
    txt2.append(f"mc_samples={int(getattr(args,'orig_mc_samples',64))}")
    txt2.append(f"score={str(args.sg_score)}")
    txt2.append("")
    txt2.append(f"coverage={coverage:.6f}")
    txt2.append(f"selective_accuracy={sel_acc:.6f}")
    txt2.append(f"avg_queries={avg_q:.6f}")
    txt2.append(f"total_evaluated={total_eval}")
    txt2.append(f"accepted={accepted}")
    txt2.append(f"correct={correct}")
    txt2.append("\nTop concepts by query frequency (k, count, avg_gain_when_queried):")
    top = np.argsort(-concept_query_count)[: min(50, CONCEPT_DIM)]
    for kk in top:
        if concept_query_count[kk] == 0:
            continue
        avg_gain = concept_gain_sum[kk] / max(1, concept_query_count[kk])
        txt2.append(f"{kk}\t{concept_query_count[kk]}\t{avg_gain:.6f}")

    _ensure_dir(out_dir)
    with open(original_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt2) + "\n")

    _append_summary_csv(
        summary_csv_path,
        {
            "tag": _sanitize_tag(run_tag),
            "method": "original_concept_sampling",
            "exp": "original",
            "ckpt": model_path,
            "dataname": args.dataname,
            "class_idxs": getattr(args, "class_idxs", "-1"),
            "overlap": overlap,
            "uncertain_policy": uncertain_policy,
            "n_uncertain_used": int(n_uncertain_used),
            "conf_thresh": float(tau),
            "budget": int(budget),
            "mc_samples": int(getattr(args, "orig_mc_samples", 64)),
            "score": str(args.sg_score),
            "coverage": float(coverage),
            "selective_accuracy": float(sel_acc),
            "avg_queries": float(avg_q),
            "n_selected_by_class": int(n_selected),
            "n_uncertain0_in_selected": int(n_uncertain0),
            "total_evaluated": int(total_eval),
            "accepted": int(accepted),
            "correct": int(correct),
            "skipped_by_class_filter": int(skipped_by_class),
        },
    )

def _make_run_suffix(run_tag: str, args, class_idx_set: Optional[set[int]]) -> str:
    cls = "all" if class_idx_set is None else "_".join(map(str, sorted(list(class_idx_set))))
    return "__".join(
        [
            f"tag-{_sanitize_tag(run_tag)}",
            f"exp-{args.exp}",
            f"overlap-{getattr(args, 'overlap', 'all')}",
            f"cls-{cls}",
            f"tau-{float(args.sg_consensus):.3f}",
            f"B-{int(args.sg_budget)}",
            f"a-{float(args.sg_alpha):.3f}",
            f"score-{args.sg_score}",
        ]
    )


def evaluate_one_run(
    run_tag: str,
    args,
    test_loader,
    device: torch.device,
    out_dir: str,
    summary_csv_path: str,
    *,
    keep_flags_override: Optional[List[bool]] = None,
    uncertain_flags_override: Optional[List[bool]] = None,
    sel_stats_override: Optional[Dict[str, int]] = None,
    uncertain_policy: str = "per_run",
) -> None:
    is_original_only = str(getattr(args, "exp", "")).lower() == "original"
    class_idx_set = _parse_class_idxs(getattr(args, "class_idxs", "-1"))
    run_suffix = _make_run_suffix(run_tag, args, class_idx_set)

    # checkpoint resolution
    model_path = getattr(args, "ckpt", "")
    if not model_path:
        log_dir = getattr(args, "log_dir", "")
        model_path = os.path.join(log_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found for run '{run_tag}': {model_path}")

    # load children
    child_models = _load_child_models_for_exp(model_path, args, device, CONCEPT_DIM, N_CLASSES)
    for m in child_models:
        m.to(device).eval()
    M = len(child_models)
    if M == 0:
        raise RuntimeError(f"No child models loaded for run '{run_tag}'")

    if uncertain_flags_override is not None and sel_stats_override is not None:
        uncertain_flags = list(uncertain_flags_override)
        sel_stats = dict(sel_stats_override)
    else:
        uncertain_flags, sel_stats = _compute_uncertain_flags(
            test_loader=test_loader,
            child_models=child_models,
            class_idx_set=class_idx_set,
            device=device,
            sg_consensus=float(getattr(args, "sg_consensus", 0.80)),
            dataname=str(getattr(args, "dataname", "CUB")),
            args=args,
        )

    overlap = getattr(args, "overlap", "all")
    if overlap == "all":
        keep_flags = [True] * len(uncertain_flags)
    elif overlap == "uncertain":
        # If main provided a shared uncertainty mask (intersection/union across runs), use it.
        keep_flags = list(keep_flags_override) if keep_flags_override is not None else list(uncertain_flags)
    else:
        raise ValueError(f"Unknown overlap='{overlap}'. Use 'all' or 'uncertain'.")

    if len(keep_flags) != len(uncertain_flags):
        raise RuntimeError(
            f"keep_flags length {len(keep_flags)} != uncertain_flags length {len(uncertain_flags)}. "
            f"This indicates a mismatch in how the sample stream was constructed."
        )

    # keep_flags already computed above (may be shared across runs via --uncertain_policy)

    n_selected = int(sel_stats["n_selected_by_class"])
    n_uncertain0 = int(sel_stats["n_uncertain0_in_selected"])
    skipped_by_class = int(sel_stats["skipped_by_class_filter"])
    n_uncertain_used = int(sum(bool(x) for x in keep_flags)) if overlap == "uncertain" else n_uncertain0

    # pass-2: evaluate on selected subset
    total_eval = 0
    accepted = 0
    correct = 0
    total_queries = 0
    concept_query_count = np.zeros(CONCEPT_DIM, dtype=np.int64)
    concept_gain_sum = np.zeros(CONCEPT_DIM, dtype=np.float64)

    # keep pointer into keep_flags (aligned with class-filtered samples)
    kptr = 0

    # helper to get concept oracle vector
    def _get_c01(c_vec) -> torch.Tensor:
        if torch.is_tensor(c_vec):
            c01 = c_vec.float()
            if c01.min().item() < 0:
                c01 = (c01 > 0).float()
            else:
                c01 = (c01 > 0.5).float()
            return c01.view(-1).to(device)
        c01 = torch.tensor(c_vec, device=device).float().view(-1)
        return (c01 > 0.5).float()

    original_txt = os.path.join(out_dir, f"original_safeguard__{run_suffix}.txt")

    if not is_original_only:
        # Member reweighting safeguard (works for any multi-child source: random seeds, LoRA Rashomon set, etc.)
        rashomon_txt = os.path.join(out_dir, f"rashomon_safeguard__{run_suffix}.txt")

        txt_lines = []
        txt_lines.append(f"Run tag: {run_tag}")
        txt_lines.append("method=member_reweighting")
        txt_lines.append(f"exp={args.exp}")
        txt_lines.append(f"ckpt={model_path}")
        txt_lines.append(f"dataname={args.dataname}")
        txt_lines.append(f"class_idxs={getattr(args,'class_idxs','-1')}")
        txt_lines.append(f"overlap={overlap}")
        txt_lines.append(f"n_selected_by_class={n_selected}")
        txt_lines.append(f"n_uncertain0_in_selected={n_uncertain0}")
        txt_lines.append(f"skipped_by_class_filter={skipped_by_class}")
        txt_lines.append("")

        with torch.no_grad():
            for data in test_loader:
                if args.dataname == "CUB":
                    inputs, y_true, c_true = data
                    x_batch = inputs.to(device) if torch.is_tensor(inputs) else inputs
                else:
                    x_batch = data["img"].to(device)
                    y_true = data["class_label"]
                    c_true = data["attribute_label"]
                B = x_batch.size(0)
                for b in range(B):
                    y = int(y_true[b].item()) if torch.is_tensor(y_true) else int(y_true[b])
                    if class_idx_set is not None and y not in class_idx_set:
                        continue
                    if kptr >= len(keep_flags):
                        continue
                    keep = keep_flags[kptr]
                    kptr += 1
                    if not keep:
                        continue

                    x = x_batch[b : b + 1]
                    c01 = _get_c01(c_true[b])

                    def oracle(k: int) -> int:
                        return int(c01[k].item())

                    out = rashomon_safeguard_predict_one(
                        x,
                        child_models,
                        num_classes=N_CLASSES,
                        expert_oracle=oracle,
                        init_feedback=None,
                        budget=int(args.sg_budget),
                        consensus_thresh=float(args.sg_consensus),
                        alpha=float(args.sg_alpha),
                        score=str(args.sg_score),
                    )
                    total_eval += 1
                    q = len(out.get("queries", []))
                    total_queries += q
                    for (kk, _vv, gain) in out.get("queries", []):
                        if 0 <= kk < CONCEPT_DIM:
                            concept_query_count[kk] += 1
                            concept_gain_sum[kk] += float(gain)

                    if out.get("accept", False):
                        accepted += 1
                        if int(out.get("yhat", -1)) == y:
                            correct += 1

        coverage = accepted / (total_eval + 1e-12)
        sel_acc = correct / (accepted + 1e-12)
        avg_q = total_queries / (total_eval + 1e-12)

        txt_lines.append("[Member safeguard]")
        txt_lines.append(f"coverage={coverage:.6f}")
        txt_lines.append(f"selective_accuracy={sel_acc:.6f}")
        txt_lines.append(f"avg_queries={avg_q:.6f}")
        txt_lines.append(f"total_evaluated={total_eval}")
        txt_lines.append(f"accepted={accepted}")
        txt_lines.append(f"correct={correct}")
        txt_lines.append("\nTop concepts by query frequency (k, count, avg_gain_when_queried):")
        top = np.argsort(-concept_query_count)[: min(50, CONCEPT_DIM)]
        for kk in top:
            if concept_query_count[kk] == 0:
                continue
            avg_gain = concept_gain_sum[kk] / max(1, concept_query_count[kk])
            txt_lines.append(f"{kk}\t{concept_query_count[kk]}\t{avg_gain:.6f}")

        _ensure_dir(out_dir)
        with open(rashomon_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + "\n")

        _append_summary_csv(
            summary_csv_path,
            {
                "tag": _sanitize_tag(run_tag),
                "method": "member_reweighting",
                "exp": args.exp,
                "ckpt": model_path,
                "dataname": args.dataname,
                "class_idxs": getattr(args, "class_idxs", "-1"),
                "overlap": overlap,
                "uncertain_policy": uncertain_policy,
                "n_uncertain_used": n_uncertain_used,
                "consensus_thresh": float(args.sg_consensus),
                "budget": int(args.sg_budget),
                "alpha": float(args.sg_alpha),
                "score": str(args.sg_score),
                "coverage": float(coverage),
                "selective_accuracy": float(sel_acc),
                "avg_queries": float(avg_q),
                "n_selected_by_class": n_selected,
                "n_uncertain0_in_selected": n_uncertain0,
                "total_evaluated": int(total_eval),
                "accepted": int(accepted),
                "correct": int(correct),
                "skipped_by_class_filter": int(skipped_by_class),
            },
        )

        print(
            f"[{run_tag}][Member] overlap={overlap} eval={total_eval} cov={coverage:.4f} "
            f"sel_acc={sel_acc:.4f} avg_q={avg_q:.2f} (selected={n_selected}, uncertain0={n_uncertain0})"
        )
        print(f"[{run_tag}] saved: {rashomon_txt}")

    # Original baseline: use member 0 as reference model
    if not getattr(args, "no_original", False):
        ref_model = child_models[0]

        # reset counters
        total_eval2 = 0
        accepted2 = 0
        correct2 = 0
        total_queries2 = 0
        concept_query_count2 = np.zeros(CONCEPT_DIM, dtype=np.int64)
        concept_gain_sum2 = np.zeros(CONCEPT_DIM, dtype=np.float64)
        kptr2 = 0

        txt2 = []
        txt2.append(f"Run tag: {run_tag}")
        txt2.append(f"baseline=original_concept_sampling")
        txt2.append(f"reference_member=0")
        txt2.append(f"exp={args.exp}")
        txt2.append(f"ckpt={model_path}")
        txt2.append(f"dataname={args.dataname}")
        txt2.append(f"class_idxs={getattr(args,'class_idxs','-1')}")
        txt2.append(f"overlap={overlap}")
        txt2.append(f"n_selected_by_class={n_selected}")
        txt2.append(f"n_uncertain0_in_selected={n_uncertain0}")
        txt2.append(f"skipped_by_class_filter={skipped_by_class}")
        txt2.append(f"conf_thresh={float(args.sg_consensus)}")
        txt2.append(f"budget={int(args.sg_budget)}")
        txt2.append(f"mc_samples={int(getattr(args,'orig_mc_samples',64))}")
        txt2.append(f"score={str(args.sg_score)}")
        txt2.append("")

        with torch.no_grad():
            for data in test_loader:
                if args.dataname == "CUB":
                    inputs, y_true, c_true = data
                    x_batch = inputs.to(device) if torch.is_tensor(inputs) else inputs
                else:
                    x_batch = data["img"].to(device)
                    y_true = data["class_label"]
                    c_true = data["attribute_label"]
                B = x_batch.size(0)
                for b in range(B):
                    y = int(y_true[b].item()) if torch.is_tensor(y_true) else int(y_true[b])
                    if class_idx_set is not None and y not in class_idx_set:
                        continue
                    if kptr2 >= len(keep_flags):
                        continue
                    keep = keep_flags[kptr2]
                    kptr2 += 1
                    if not keep:
                        continue

                    x = x_batch[b : b + 1]
                    c01 = _get_c01(c_true[b])
                    def oracle(k: int) -> int:
                        return int(c01[k].item())

                    out = original_safeguard_predict_one(
                        x,
                        ref_model,
                        num_classes=N_CLASSES,
                        expert_oracle=oracle,
                        init_feedback=None,
                        budget=int(args.sg_budget),
                        conf_thresh=float(args.sg_consensus),
                        mc_samples=int(getattr(args, "orig_mc_samples", 64)),
                        score=str(args.sg_score),
                    )

                    total_eval2 += 1
                    q = len(out.get("queries", []))
                    total_queries2 += q
                    for (kk, _vv, gain) in out.get("queries", []):
                        if 0 <= kk < CONCEPT_DIM:
                            concept_query_count2[kk] += 1
                            concept_gain_sum2[kk] += float(gain)

                    if out.get("accept", False):
                        accepted2 += 1
                        if int(out.get("yhat", -1)) == y:
                            correct2 += 1

        coverage2 = accepted2 / (total_eval2 + 1e-12)
        sel_acc2 = correct2 / (accepted2 + 1e-12)
        avg_q2 = total_queries2 / (total_eval2 + 1e-12)

        txt2.append(f"coverage={coverage2:.6f}")
        txt2.append(f"selective_accuracy={sel_acc2:.6f}")
        txt2.append(f"avg_queries={avg_q2:.6f}")
        txt2.append(f"total_evaluated={total_eval2}")
        txt2.append(f"accepted={accepted2}")
        txt2.append(f"correct={correct2}")
        txt2.append("\nTop concepts by query frequency (k, count, avg_gain_when_queried):")
        top = np.argsort(-concept_query_count2)[: min(50, CONCEPT_DIM)]
        for kk in top:
            if concept_query_count2[kk] == 0:
                continue
            avg_gain = concept_gain_sum2[kk] / max(1, concept_query_count2[kk])
            txt2.append(f"{kk}\t{concept_query_count2[kk]}\t{avg_gain:.6f}")

        with open(original_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(txt2) + "\n")

        _append_summary_csv(
            summary_csv_path,
            {
                "tag": _sanitize_tag(run_tag),
                "method": "original_concept_sampling",
                "exp": args.exp,
                "ckpt": model_path,
                "dataname": args.dataname,
                "class_idxs": getattr(args, "class_idxs", "-1"),
                "overlap": overlap,
                "uncertain_policy": uncertain_policy,
                "n_uncertain_used": n_uncertain_used,
                "conf_thresh": float(args.sg_consensus),
                "budget": int(args.sg_budget),
                "mc_samples": int(getattr(args, "orig_mc_samples", 64)),
                "score": str(args.sg_score),
                "coverage": float(coverage2),
                "selective_accuracy": float(sel_acc2),
                "avg_queries": float(avg_q2),
                "n_selected_by_class": n_selected,
                "n_uncertain0_in_selected": n_uncertain0,
                "total_evaluated": int(total_eval2),
                "accepted": int(accepted2),
                "correct": int(correct2),
                "skipped_by_class_filter": int(skipped_by_class),
            },
        )

        print(
            f"[{run_tag}][Original] overlap={overlap} eval={total_eval2} cov={coverage2:.4f} "
            f"sel_acc={sel_acc2:.4f} avg_q={avg_q2:.2f} (selected={n_selected}, uncertain0={n_uncertain0})"
        )
        print(f"[{run_tag}] saved: {original_txt}")

def _cache_member_run_trajectory_for_sweep(
    *,
    run_tag: str,
    args,
    test_loader,
    device: torch.device,
    budget_max: int,
    stop_tau: Optional[float] = None,
    progress_every: int = 250,
) -> Dict[str, object]:
    """Load member models once and cache their per-sample trajectories up to budget_max.

    Note: this caches only what is needed for sweeps (consensus0 + query trajectory), not the raw images.
    """
    class_idx_set = _parse_class_idxs(getattr(args, "class_idxs", "-1"))

    model_path = getattr(args, "ckpt", "")
    if not model_path:
        log_dir = getattr(args, "log_dir", "")
        model_path = os.path.join(log_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found for run '{run_tag}': {model_path}")

    child_models = _load_child_models_for_exp(model_path, args, device, CONCEPT_DIM, N_CLASSES)
    for m in child_models:
        m.to(device).eval()
    M = len(child_models)
    if M == 0:
        raise RuntimeError(f"No child models loaded for run '{run_tag}'")

    y_true_list: List[int] = []
    cons0_list: List[float] = []
    traj_cons_seq: List[List[float]] = []
    traj_yhat_seq: List[List[int]] = []
    traj_queries: List[List[Tuple[int, int, float]]] = []

    n_selected = 0
    skipped_by_class = 0

    t0 = time.time()
    last_print = t0

    with torch.no_grad():
        for data in test_loader:
            if str(getattr(args, "dataname", "CUB")) == "CUB":
                inputs, y_true, c_true = data
                x_batch = inputs.to(device) if torch.is_tensor(inputs) else inputs
            else:
                x_batch = data["img"].to(device)
                y_true = data["class_label"]
                c_true = data.get("attribute_label", None)

            B = x_batch.size(0)

            # Member forward for the whole batch
            y_logits_list = []
            c_logits_list = []
            any_missing_concepts = False
            for m in child_models:
                y_m, c_m = _member_forward(m, x_batch, num_classes=N_CLASSES)
                y_logits_list.append(y_m)
                c_cat = _concat_concepts(c_m)
                if c_cat is None:
                    any_missing_concepts = True
                c_logits_list.append(c_cat)

            y_logits_MBC = torch.stack(y_logits_list, dim=0)  # [M,B,C]
            y_probs_MBC = _as_prob_multiclass(y_logits_MBC)

            if any_missing_concepts:
                c_logits_MBK = None
            else:
                c_logits_MBK = torch.stack(c_logits_list, dim=0)  # [M,B,K]

            # class filter indices for this batch
            keep_bs: List[int] = []
            for b in range(B):
                yb = int(y_true[b].item()) if torch.is_tensor(y_true) else int(y_true[b])
                if class_idx_set is not None and yb not in class_idx_set:
                    skipped_by_class += 1
                    continue
                keep_bs.append(b)
                y_true_list.append(yb)
                n_selected += 1

            if not keep_bs:
                continue

            consB = _uniform_consensus_batch(y_probs_MBC[:, keep_bs, :])  # [Kb] on CPU
            for j, b in enumerate(keep_bs):
                cons0 = float(consB[j].item())
                cons0_list.append(cons0)

                if c_true is None:
                    c01 = None
                else:
                    c01 = c_true[b]
                    if torch.is_tensor(c01):
                        c01 = c01.float()
                        if c01.min().item() < 0:
                            c01 = (c01 > 0).float()
                        else:
                            c01 = (c01 > 0.5).float()
                        c01 = c01.view(-1).cpu()
                    else:
                        c01 = torch.tensor(c01).float().view(-1)
                        c01 = (c01 > 0.5).float()

                y_probs_M = y_probs_MBC[:, b, :].detach().cpu()
                if c_logits_MBK is None:
                    c_logits_MK = None
                else:
                    c_logits_MK = c_logits_MBK[:, b, :].detach().cpu()

                traj = _member_trajectory_from_logits(
                    y_probs_M=y_probs_M.to(dtype=torch.float32),
                    c_logits_MK=(c_logits_MK.to(dtype=torch.float32) if c_logits_MK is not None else None),
                    c_oracle01=c01,
                    budget_max=int(budget_max),
                    alpha=float(getattr(args, "sg_alpha", 5.0)),
                    score=str(getattr(args, "sg_score", "entropy")),
                    stop_tau=stop_tau,
                )
                traj_cons_seq.append(list(traj["cons_seq"]))
                traj_yhat_seq.append(list(traj["yhat_seq"]))
                traj_queries.append(list(traj["queries"]))

                if progress_every > 0 and (n_selected % int(progress_every) == 0):
                    now = time.time()
                    if now - last_print >= 1.0:
                        rate = n_selected / max(1e-9, now - t0)
                        print(f"[SweepCache][{run_tag}][Member] cached={n_selected} ({rate:.2f} samp/s)")
                        last_print = now

    # cleanup GPU memory
    del child_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y_true_N = torch.tensor(y_true_list, dtype=torch.long)
    cons0_N = torch.tensor(cons0_list, dtype=torch.float32)

    return {
        "run_tag": run_tag,
        "model_path": model_path,
        "y_true_N": y_true_N,
        "cons0_N": cons0_N,
        "traj_cons_seq": traj_cons_seq,
        "traj_yhat_seq": traj_yhat_seq,
        "traj_queries": traj_queries,
        "sel_stats_base": {
            "n_selected_by_class": int(n_selected),
            "skipped_by_class_filter": int(skipped_by_class),
        },
    }


def evaluate_sweep(
    *,
    args,
    test_loader,
    device: torch.device,
    out_dir: str,
    summary_csv: str,
) -> None:
    """Sweep sg_consensus (tau) and sg_budget (B) in a single process to avoid repeated I/O.

    Supports member-based methods (random / LoRA / DivEns / X2C / etc.) and the original
    MC concept-sampling baseline (exp='original').
    """
    tau_list = _parse_csv_floats(getattr(args, "sweep_taus", ""))
    budget_list = _parse_csv_ints(getattr(args, "sweep_budgets", ""))
    if not tau_list:
        tau_list = [float(getattr(args, "sg_consensus", 0.80))]
    if not budget_list:
        budget_list = [int(getattr(args, "sg_budget", 8))]
    budget_max = max(budget_list) if budget_list else int(getattr(args, "sg_budget", 8))
    stop_tau = max(tau_list) if tau_list else None
    class_sweep = _parse_class_sweep(getattr(args, "sweep_class_idxs", ""))
    class_specs = class_sweep if class_sweep else [str(getattr(args, "class_idxs", "-1"))]

    overlap = str(getattr(args, "overlap", "all"))
    policy = str(getattr(args, "uncertain_policy", "per_run"))

    if args.runs_json:
        runs = _load_runs_from_json(args.runs_json)
        base_args = copy.deepcopy(args)

        original_runs = [r for r in runs if str(r.get("exp", "")).lower() == "original"]
        if len(original_runs) > 1:
            raise ValueError("[Sweep] runs_json contains more than one entry with exp='original'. Keep only one.")
        original_run = None if getattr(args, "no_original", False) else (original_runs[0] if original_runs else None)

        eval_runs = [r for r in runs if str(r.get("exp", "")).lower() != "original"]
        if not eval_runs and original_run is None:
            raise ValueError("[Sweep] No runs to evaluate (all runs are original but --no_original is set).")

        caches: Dict[str, Dict[str, object]] = {}
        run_args: Dict[str, object] = {}

        for i, run in enumerate(eval_runs):
            args_i = copy.deepcopy(base_args)
            _apply_run_overrides(args_i, run)
            if class_sweep:
                # Cache once over all classes; filter per-class during eval.
                args_i.class_idxs = "-1"
            run_tag = str(run.get("tag", run.get("run_tag", f"run{i+1}")))
            args_i.run_tag = run_tag
            run_args[run_tag] = args_i
            caches[run_tag] = _cache_member_run_trajectory_for_sweep(
                run_tag=run_tag,
                args=args_i,
                test_loader=test_loader,
                device=device,
                budget_max=budget_max,
                stop_tau=stop_tau,
            )

        tags = list(caches.keys())
        original_cache = None
        original_args = None
        original_tag = None
        if original_run is not None:
            original_args = copy.deepcopy(base_args)
            _apply_run_overrides(original_args, original_run)
            original_tag = str(original_run.get("tag", original_run.get("run_tag", "original")))
            original_args.run_tag = original_tag
            # Force exp to original for caching/eval
            original_args.exp = "original"
            if class_sweep:
                original_args.class_idxs = "-1"
            original_cache = _cache_original_run_trajectory_for_sweep(
                run_tag=original_tag,
                args=original_args,
                test_loader=test_loader,
                device=device,
                budget_max=budget_max,
                stop_tau=stop_tau,
            )

        # Determine shared stream reference for alignment checks.
        if tags:
            ref = caches[tags[0]]
            N = int(ref["sel_stats_base"]["n_selected_by_class"])
            y_ref = ref["y_true_N"]
            for t in tags[1:]:
                c = caches[t]
                if int(c["sel_stats_base"]["n_selected_by_class"]) != N:
                    raise ValueError(
                        f"[Sweep] selected-stream length mismatch: {tags[0]}={N} vs {t}={int(c['sel_stats_base']['n_selected_by_class'])}"
                    )
                if not torch.equal(y_ref, c["y_true_N"]):
                    raise ValueError(f"[Sweep] y_true stream mismatch between {tags[0]} and {t}.")
            if original_cache is not None:
                if int(original_cache["sel_stats_base"]["n_selected_by_class"]) != N:
                    raise ValueError(
                        f"[Sweep] selected-stream length mismatch: {tags[0]}={N} vs {original_tag}={int(original_cache['sel_stats_base']['n_selected_by_class'])}"
                    )
                if not torch.equal(y_ref, original_cache["y_true_N"]):
                    raise ValueError(f"[Sweep] y_true stream mismatch between {tags[0]} and {original_tag}.")
        else:
            # Only original run in JSON
            assert original_cache is not None
            N = int(original_cache["sel_stats_base"]["n_selected_by_class"])

        # Stable y_true stream for class filtering.
        y_stream = caches[tags[0]]["y_true_N"] if tags else original_cache["y_true_N"]
        y_stream_list = [int(x) for x in y_stream.tolist()]

        for class_spec in class_specs:
            class_idx_set = _parse_class_idxs(class_spec)
            sel_mask = [True] * N if class_idx_set is None else [y in class_idx_set for y in y_stream_list]
            n_selected_cls = int(sum(bool(x) for x in sel_mask))
            skipped_cls = int(N - n_selected_cls)

            for tau in tau_list:
                flags_by_run: Dict[str, List[bool]] = {}
                n_unc0_by_run: Dict[str, int] = {}
                for t in tags:
                    cons0 = caches[t]["cons0_N"]
                    raw = (cons0 < float(tau)).tolist()
                    f = [bool(raw[j]) and bool(sel_mask[j]) for j in range(N)]
                    flags_by_run[t] = f
                    n_unc0_by_run[t] = int(sum(bool(x) for x in f))

                shared_keep_flags: Optional[List[bool]] = None
                if overlap == "uncertain" and policy in {"intersection", "union"} and len(tags) >= 2:
                    if policy == "intersection":
                        shared_keep_flags = [bool(sel_mask[j]) and all(flags_by_run[t][j] for t in tags) for j in range(N)]
                    else:
                        shared_keep_flags = [bool(sel_mask[j]) and any(flags_by_run[t][j] for t in tags) for j in range(N)]

                for b in budget_list:
                    for t in tags:
                        args_i = copy.deepcopy(run_args[t])
                        args_i.sg_consensus = float(tau)
                        args_i.sg_budget = int(b)
                        args_i.class_idxs = str(class_spec)

                        if overlap == "all":
                            keep_flags = list(sel_mask)
                        else:
                            keep_flags = list(shared_keep_flags) if shared_keep_flags is not None else list(flags_by_run[t])

                        sel_stats = {
                            "n_selected_by_class": int(n_selected_cls),
                            "n_uncertain0_in_selected": int(n_unc0_by_run[t]),
                            "skipped_by_class_filter": int(skipped_cls),
                        }

                        _eval_member_from_cached_trajectory(
                            run_tag=t,
                            args=args_i,
                            out_dir=out_dir,
                            summary_csv_path=summary_csv,
                            model_path=str(caches[t]["model_path"]),
                            y_true_N=caches[t]["y_true_N"],
                            cons0_N=caches[t]["cons0_N"],
                            traj_cons_seq=caches[t]["traj_cons_seq"],
                            traj_yhat_seq=caches[t]["traj_yhat_seq"],
                            traj_queries=caches[t]["traj_queries"],
                            keep_flags=keep_flags,
                            sel_stats=sel_stats,
                            uncertain_policy=policy,
                        )

                    # Evaluate original baseline (MC) once per (tau, B) on the same keep_flags semantics.
                    if original_cache is not None and original_args is not None and original_tag is not None:
                        args_o = copy.deepcopy(original_args)
                        args_o.sg_consensus = float(tau)
                        args_o.sg_budget = int(b)
                        args_o.class_idxs = str(class_spec)

                        conf0 = original_cache["conf0_N"]
                        raw0 = (conf0 < float(tau)).tolist()
                        f0 = [bool(raw0[j]) and bool(sel_mask[j]) for j in range(N)]
                        n_unc0 = int(sum(bool(x) for x in f0))

                        if overlap == "all":
                            keep_o = list(sel_mask)
                        else:
                            keep_o = list(shared_keep_flags) if shared_keep_flags is not None else list(f0)

                        sel_stats_o = {
                            "n_selected_by_class": int(n_selected_cls),
                            "n_uncertain0_in_selected": int(n_unc0),
                            "skipped_by_class_filter": int(skipped_cls),
                        }

                        _eval_original_from_cached_trajectory(
                            run_tag=original_tag,
                            args=args_o,
                            out_dir=out_dir,
                            summary_csv_path=summary_csv,
                            model_path=str(original_cache["model_path"]),
                            y_true_N=original_cache["y_true_N"],
                            conf0_N=original_cache["conf0_N"],
                            traj_conf_seq=original_cache["traj_conf_seq"],
                            traj_yhat_seq=original_cache["traj_yhat_seq"],
                            traj_queries=original_cache["traj_queries"],
                            keep_flags=keep_o,
                            sel_stats=sel_stats_o,
                            uncertain_policy=policy,
                        )
        return

    # Single-run sweep
    run_tag = args.tag or args.exp
    if str(getattr(args, "exp", "")).lower() == "original":
        if getattr(args, "no_original", False):
            raise ValueError("[Sweep] exp='original' but --no_original is set.")
        cache_args = copy.deepcopy(args)
        if class_sweep:
            cache_args.class_idxs = "-1"
        cache_o = _cache_original_run_trajectory_for_sweep(
            run_tag=run_tag,
            args=cache_args,
            test_loader=test_loader,
            device=device,
            budget_max=budget_max,
            stop_tau=stop_tau,
        )
        N = int(cache_o["sel_stats_base"]["n_selected_by_class"])
        conf0 = cache_o["conf0_N"]
        y_stream_list = [int(x) for x in cache_o["y_true_N"].tolist()]
        for class_spec in class_specs:
            class_idx_set = _parse_class_idxs(class_spec)
            sel_mask = [True] * N if class_idx_set is None else [y in class_idx_set for y in y_stream_list]
            n_selected_cls = int(sum(bool(x) for x in sel_mask))
            skipped_cls = int(N - n_selected_cls)
            for tau in tau_list:
                raw0 = (conf0 < float(tau)).tolist()
                flags = [bool(raw0[j]) and bool(sel_mask[j]) for j in range(N)]
                n_unc0 = int(sum(bool(x) for x in flags))
                for b in budget_list:
                    args_i = copy.deepcopy(args)
                    args_i.sg_consensus = float(tau)
                    args_i.sg_budget = int(b)
                    args_i.class_idxs = str(class_spec)
                    keep_flags = list(sel_mask) if overlap == "all" else list(flags)
                    sel_stats = {
                        "n_selected_by_class": int(n_selected_cls),
                        "n_uncertain0_in_selected": int(n_unc0),
                        "skipped_by_class_filter": int(skipped_cls),
                    }
                    _eval_original_from_cached_trajectory(
                        run_tag=run_tag,
                        args=args_i,
                        out_dir=out_dir,
                        summary_csv_path=summary_csv,
                        model_path=str(cache_o["model_path"]),
                        y_true_N=cache_o["y_true_N"],
                        conf0_N=cache_o["conf0_N"],
                        traj_conf_seq=cache_o["traj_conf_seq"],
                        traj_yhat_seq=cache_o["traj_yhat_seq"],
                        traj_queries=cache_o["traj_queries"],
                        keep_flags=keep_flags,
                        sel_stats=sel_stats,
                        uncertain_policy=policy,
                    )
        return

    cache_args = copy.deepcopy(args)
    if class_sweep:
        cache_args.class_idxs = "-1"
    cache = _cache_member_run_trajectory_for_sweep(
        run_tag=run_tag,
        args=cache_args,
        test_loader=test_loader,
        device=device,
        budget_max=budget_max,
        stop_tau=stop_tau,
    )
    N = int(cache["sel_stats_base"]["n_selected_by_class"])
    cons0 = cache["cons0_N"]
    y_stream_list = [int(x) for x in cache["y_true_N"].tolist()]
    for class_spec in class_specs:
        class_idx_set = _parse_class_idxs(class_spec)
        sel_mask = [True] * N if class_idx_set is None else [y in class_idx_set for y in y_stream_list]
        n_selected_cls = int(sum(bool(x) for x in sel_mask))
        skipped_cls = int(N - n_selected_cls)
        for tau in tau_list:
            raw = (cons0 < float(tau)).tolist()
            flags = [bool(raw[j]) and bool(sel_mask[j]) for j in range(N)]
            n_unc0 = int(sum(bool(x) for x in flags))
            for b in budget_list:
                args_i = copy.deepcopy(args)
                args_i.sg_consensus = float(tau)
                args_i.sg_budget = int(b)
                args_i.class_idxs = str(class_spec)
                keep_flags = list(sel_mask) if overlap == "all" else list(flags)
                sel_stats = {
                    "n_selected_by_class": int(n_selected_cls),
                    "n_uncertain0_in_selected": int(n_unc0),
                    "skipped_by_class_filter": int(skipped_cls),
                }
                _eval_member_from_cached_trajectory(
                    run_tag=run_tag,
                    args=args_i,
                    out_dir=out_dir,
                    summary_csv_path=summary_csv,
                    model_path=str(cache["model_path"]),
                    y_true_N=cache["y_true_N"],
                    cons0_N=cache["cons0_N"],
                    traj_cons_seq=cache["traj_cons_seq"],
                    traj_yhat_seq=cache["traj_yhat_seq"],
                    traj_queries=cache["traj_queries"],
                    keep_flags=keep_flags,
                    sel_stats=sel_stats,
                    uncertain_policy=policy,
                )


# =======================================================
# CLI
# =======================================================

def parse_args():
    p = argparse.ArgumentParser("Rashomon/Random/Original safeguard evaluation (multi-run)")

    # core model selection (single-run mode)
    p.add_argument(
        "--exp",
        type=str,
        default="Lora",
        choices=["DivEns", "random", "ConvAda", "Lora", "Dropout", "original"],
        help="Experiment type / architecture family.",
    )
    p.add_argument("--log_dir", type=str, default="", help="Path to exp log directory (for best_model.pth).")
    p.add_argument("--ckpt", type=str, default="", help="Direct path to checkpoint (overrides log_dir/best_model.pth).")

    # multi-run mode
    p.add_argument("--runs_json", type=str, default="", help="JSON with list of runs (each overrides args).")
    p.add_argument("--tag", type=str, default="", help="Single-run tag for filenames/CSV.")
    p.add_argument("--run_tag", dest="tag", type=str, default="", help="Alias of --tag (backward compatible).")
    p.add_argument("--out_dir", type=str, default="", help="Output dir for txt + summary CSV.")
    p.add_argument("--summary_csv", type=str, default="", help="Summary CSV path (append).")

    # data
    p.add_argument("--dataname", type=str, default="CUB", choices=["CUB", "cifar10", "Awa2", "CelebA", "HAM10000"])
    p.add_argument("--data_dir", type=str, default="cifar10_data", help="Dataset folder (relative to BASE_DIR unless abs)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--eval_splits",
        type=str,
        default="test",
        help="Which dataset split(s) to evaluate on. Options: train,val,test or comma-separated like 'val,test' or 'train,val,test'. "
             "Note: using 'train' removes the held-out guarantee and can inflate results.",
    )

    # selection
    p.add_argument("--class_idxs", type=str, default="-1", help='Class indices to evaluate; "-1" means all.')
    p.add_argument("--overlap", type=str, default="all", choices=["all", "uncertain"], help="Eval subset selector")
    p.add_argument(
        "--uncertain_policy",
        type=str,
        default="per_run",
        choices=["per_run", "intersection", "union"],
        help="When --overlap=uncertain and --runs_json is used, how to define the uncertain subset across runs. "
             "'per_run' uses each run's own uncertain set; 'intersection' uses samples uncertain for all runs; "
             "'union' uses samples uncertain for at least one run.",
    )
    p.add_argument(
        "--orig_mode",
        type=str,
        default="mc",
        choices=["mc"],
        help="Original safeguard baseline uses concept MC sampling (only).",
    )

    # safeguard hyperparams
    p.add_argument("--sg_consensus", type=float, default=0.80)
    p.add_argument("--sg_budget", type=int, default=8)
    p.add_argument("--sg_alpha", type=float, default=5.0)
    p.add_argument("--sg_score", type=str, default="entropy", choices=["entropy", "1-consensus"])

    # sweep mode (avoid repeated I/O when sweeping tau/budget)
    p.add_argument(
        "--sweep_taus",
        type=str,
        default="",
        help="Comma/space-separated list of sg_consensus values to sweep (e.g. '0.6,0.7,0.8'). "
             "If empty, uses --sg_consensus.",
    )
    p.add_argument(
        "--sweep_budgets",
        type=str,
        default="",
        help="Comma/space-separated list of sg_budget values to sweep (e.g. '0,1,2,3,4,5,6,7'). "
             "If empty, uses --sg_budget.",
    )
    p.add_argument(
        "--sweep_class_idxs",
        type=str,
        default="",
        help="Optional sweep over class_idxs without reloading models/data. "
             "Use ';' (recommended) or '|' to separate variants, e.g. '-1;0;1;2;3;4;5;6' or '0,1;2,3'. "
             "If empty, uses --class_idxs.",
    )

    # model reconstruction
    p.add_argument("--num_models", type=int, default=3)
    p.add_argument("--bottleneck_dim", type=int, default=64)
    p.add_argument("--share_mask", type=str, default="000000000000")
    p.add_argument("--encoder", type=str, default="vit")
    p.add_argument("--expand_dim", type=int, default=0)
    p.add_argument("--n_class_attr", type=int, default=CUB_N_CLASSES)
    p.add_argument("--use_aux", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    # original baseline
    p.add_argument("--orig_mc_samples", type=int, default=64)
    p.add_argument("--no_original", action="store_true")

    args = p.parse_args()
    # Backward-compatible aliases
    if not hasattr(args, 'run_tag'):
        setattr(args, 'run_tag', getattr(args, 'tag', ''))
    if not hasattr(args, 'tag'):
        setattr(args, 'tag', getattr(args, 'run_tag', ''))
    return args


def _apply_run_overrides(args, run: Dict[str, object]) -> None:
    """Override argparse namespace fields from a run dict."""
    for k, v in run.items():
        if k in ("tag", "run_tag"):
            setattr(args, "tag", str(v))
        else:
            setattr(args, k, v)


def main():
    global N_CLASSES, CONCEPT_DIM
    args = parse_args()

    # dataset meta
    if args.dataname == "CUB":
        N_CLASSES = CUB_N_CLASSES
        CONCEPT_DIM = 112
    elif args.dataname == "Awa2":
        N_CLASSES = AWA2_N_CLASSES
        CONCEPT_DIM = 85
    elif args.dataname == "cifar10":
        N_CLASSES = CIFAR10_N_CLASSES
        CONCEPT_DIM = 143
    elif args.dataname == "CelebA":
        N_CLASSES = CELEBA_N_CLASSES
        CONCEPT_DIM = 6
    elif args.dataname == "HAM10000":
        N_CLASSES = HAM10000_N_CLASSES
        CONCEPT_DIM = 139
    else:
        raise ValueError(f"Unknown dataname: {args.dataname}")

    # load data once
    _train_loader, _val_loader, test_loader = _load_split_loaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output paths
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.dirname(args.runs_json) if args.runs_json else (args.log_dir or ".")
    _ensure_dir(out_dir)
    summary_csv = args.summary_csv or os.path.join(out_dir, "safeguard_summary.csv")

    # Sweep tau/budget/class_idxs in a single process to avoid repeated model/data I/O.
    if (
        str(getattr(args, "sweep_taus", "")).strip()
        or str(getattr(args, "sweep_budgets", "")).strip()
        or str(getattr(args, "sweep_class_idxs", "")).strip()
    ):
        evaluate_sweep(args=args, test_loader=test_loader, device=device, out_dir=out_dir, summary_csv=summary_csv)
        return

    # multi-run
    if args.runs_json:
        runs = _load_runs_from_json(args.runs_json)
        base_args = copy.deepcopy(args)

        # Separate out a single-model "original" run, if present in runs_json.
        original_runs = [r for r in runs if str(r.get("exp", "")).lower() == "original"]
        if len(original_runs) > 1:
            raise ValueError("runs_json contains more than one entry with exp='original'. Keep only one.")
        original_run = original_runs[0] if original_runs else None
        eval_runs = [r for r in runs if r is not original_run]

        # Optional: define a *shared* uncertain subset across runs (intersection / union).
        shared_keep_flags: Optional[List[bool]] = None
        pre_uncertain_flags: Dict[str, List[bool]] = {}
        pre_sel_stats: Dict[str, Dict[str, int]] = {}
        shared_uncertain_used: Optional[int] = None

        overlap = getattr(args, "overlap", "all")
        policy = str(getattr(args, "uncertain_policy", "per_run"))

        if overlap == "uncertain" and policy in {"intersection", "union"} and len(eval_runs) >= 2:
            # Precompute each run's uncertain0 flags (budget=0 abstain set) on the same class-filtered stream.
            flags_list: List[List[bool]] = []
            tags: List[str] = []

            for i, run in enumerate(eval_runs):
                args_i = copy.deepcopy(base_args)
                _apply_run_overrides(args_i, run)
                run_tag = str(run.get("tag", run.get("run_tag", f"run{i+1}")))
                args_i.run_tag = run_tag

                # Load models just for uncertain-set construction
                model_path = getattr(args_i, 'ckpt', None) or os.path.join(str(args_i.log_dir), 'best_model.pth')
                child_models = _load_child_models_for_exp(model_path, args_i, device, CONCEPT_DIM, N_CLASSES)
                for cm in child_models:
                    cm.eval()

                uncertain_flags_i, sel_stats_i = _compute_uncertain_flags(
                    test_loader=test_loader,
                    args=args_i,
                    device=device,
                    child_models=child_models,
                    class_idx_set=_parse_class_idxs(getattr(args_i, "class_idxs", "-1")),
                    sg_consensus=float(getattr(args_i, "sg_consensus", 0.80)),
                    dataname=str(getattr(args_i, "dataname", "CUB")),
                )

                pre_uncertain_flags[run_tag] = list(uncertain_flags_i)
                pre_sel_stats[run_tag] = dict(sel_stats_i)
                flags_list.append(list(uncertain_flags_i))
                tags.append(run_tag)

                # cleanup
                del child_models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            L = len(flags_list[0])
            if any(len(f) != L for f in flags_list):
                lens = {tags[j]: len(flags_list[j]) for j in range(len(tags))}
                raise ValueError(f"Cannot combine uncertain sets: inconsistent selected-stream lengths: {lens}")

            if policy == "intersection":
                shared_keep_flags = [all(f[j] for f in flags_list) for j in range(L)]
            else:
                shared_keep_flags = [any(f[j] for f in flags_list) for j in range(L)]

            shared_uncertain_used = int(sum(bool(x) for x in shared_keep_flags))

        # Now evaluate each run, optionally using the shared_keep_flags.
        for i, run in enumerate(eval_runs):
            args_i = copy.deepcopy(base_args)
            _apply_run_overrides(args_i, run)
            run_tag = str(run.get("tag", run.get("run_tag", f"run{i+1}")))
            args_i.run_tag = run_tag
            # If an "original" run is provided in runs_json, skip per-run original baselines;
            # we will evaluate the original baseline on that separate model once.
            if original_run is not None:
                args_i.no_original = True

            keep_override = None
            uncertain_override = None
            stats_override = None

            if shared_keep_flags is not None:
                keep_override = shared_keep_flags
                uncertain_override = pre_uncertain_flags.get(run_tag)
                stats_override = pre_sel_stats.get(run_tag)

            evaluate_one_run(
                run_tag=run_tag,
                args=args_i,
                test_loader=test_loader,
                device=device,
                out_dir=out_dir,
                summary_csv_path=summary_csv,
                keep_flags_override=keep_override,
                uncertain_flags_override=uncertain_override,
                sel_stats_override=stats_override,
                uncertain_policy=policy,
            )

        # Evaluate the original (single-model) baseline once, if provided.
        if original_run is not None:
            args_o = copy.deepcopy(base_args)
            _apply_run_overrides(args_o, original_run)
            run_tag_o = str(original_run.get("tag", original_run.get("run_tag", "original")))
            args_o.run_tag = run_tag_o
            args_o.exp = "original"
            # Run original baseline only (no member-reweighting).
            evaluate_one_run(
                run_tag=run_tag_o,
                args=args_o,
                test_loader=test_loader,
                device=device,
                out_dir=out_dir,
                summary_csv_path=summary_csv,
                keep_flags_override=shared_keep_flags,
                uncertain_flags_override=None,
                sel_stats_override=None,
                uncertain_policy=policy,
            )

        return

    # single-run

    run_tag = args.tag or args.exp
    evaluate_one_run(run_tag, args, test_loader, device, out_dir, summary_csv)


if __name__ == "__main__":
    main()
