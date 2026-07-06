import os
import re
import json
import pickle
import shutil
import zipfile
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor


HAM7 = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM7_TO_ID = {k: i for i, k in enumerate(HAM7)}

TOPKEY_TO_HAM7 = {
    "actinic keratoses": "akiec",
    "basal cell carcinoma": "bcc",
    "benign keratosis-like lesions": "bkl",
    "dermatofibroma": "df",
    "melanocytic nevi": "nv",
    "melanoma": "mel",
    "vascular lesions": "vasc",
}

ADACBM_HAM10000_JSON_RAW = "" #todo change to your own path

DV_DOI = "doi:10.7910/DVN/DBW86T"
DV_DATASET_ZIP_URL = "" #todo change to your own path
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def wget_download(url: str, out_path: str, resume: bool = False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["wget"]
    if resume:
        cmd += ["-c"]
    cmd += ["-O", out_path, url]
    _run(cmd)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")
    return out_path
    
def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (code={p.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )
    return p

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def wget_download(url: str, out_path: str, resume: bool = False):
    _ensure_dir(os.path.dirname(out_path))
    cmd = ["wget"]
    if resume:
        cmd += ["-c"]
    cmd += ["-O", out_path, url]
    _run(cmd)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Downloaded file is empty: {out_path}")
    return out_path

def _safe_clear_dir(d):
    if not os.path.isdir(d):
        return
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)

def _unzip(zip_path: str, extract_to: str):
    _ensure_dir(extract_to)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

def _count_images(images_root):
    n = 0
    for _, _, files in os.walk(images_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                n += 1
    return n

def _flatten_images(src_root: str, dst_root: str):
    _ensure_dir(dst_root)
    moved = 0
    for dirpath, _, files in os.walk(src_root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMG_EXTS:
                continue
            src = os.path.join(dirpath, fn)
            dst = os.path.join(dst_root, fn)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1
    return moved

def _find_first_file(root, pattern, exts=None):
    rx = re.compile(pattern, re.IGNORECASE)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if exts is not None and ext not in exts:
                continue
            if rx.search(fn):
                return os.path.join(dirpath, fn)
    return None

def _find_all_files(root, pattern, exts=None):
    rx = re.compile(pattern, re.IGNORECASE)
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if exts is not None and ext not in exts:
                continue
            if rx.search(fn):
                out.append(os.path.join(dirpath, fn))
    return out

def _sniff_sep(path: str):
    with open(path, "rb") as f:
        head = f.read(4096)
    try:
        s = head.decode("utf-8", errors="ignore")
    except Exception:
        s = str(head)
    first = s.splitlines()[0] if s.splitlines() else s
    if "\t" in first:
        return "\t"
    if "," in first:
        return ","
    if ";" in first:
        return ";"
    return None

    

def ensure_ham10000_raw(out_dir: str):
    out_dir = os.path.abspath(out_dir)
    images_root = os.path.join(out_dir, "images")
    meta_csv = os.path.join(out_dir, "HAM10000_metadata.csv")

    if os.path.exists(meta_csv) and os.path.isdir(images_root) and _count_images(images_root) > 0:
        print(f"[skip] Raw HAM10000 exists: {meta_csv} and {images_root}")
        return images_root, meta_csv

    raw_dir = _ensure_dir(os.path.join(out_dir, "raw_dataverse"))
    bundle_zip = os.path.join(raw_dir, "ham10000_bundle.zip")
    bundle_extract = _ensure_dir(os.path.join(raw_dir, "bundle_extracted"))
    images_extract = _ensure_dir(os.path.join(raw_dir, "images_extracted"))

    if not (os.path.exists(bundle_zip) and os.path.getsize(bundle_zip) > 0):
        print(f"[wget] {DV_DATASET_ZIP_URL}")
        wget_download(DV_DATASET_ZIP_URL, bundle_zip, resume=True)

    _safe_clear_dir(bundle_extract)
    _safe_clear_dir(images_extract)

    _unzip(bundle_zip, bundle_extract)

    meta_path = _find_first_file(bundle_extract, r"^ham10000_metadata$", exts={".csv", ".tab", ".tsv", ""})
    if meta_path is None:
        meta_path = _find_first_file(bundle_extract, r"ham10000.*meta", exts={".csv", ".tab", ".tsv", ""})
    if meta_path is None:
        meta_path = _find_first_file(bundle_extract, r"metadata", exts={".csv", ".tab", ".tsv", ""})
    if meta_path is None:
        raise RuntimeError(f"Cannot find metadata file inside bundle: {bundle_zip}")

    ext = os.path.splitext(meta_path)[1].lower()
    if ext == ".csv":
        shutil.copyfile(meta_path, meta_csv)
    else:
        sep = _sniff_sep(meta_path)
        if sep is not None:
            df = pd.read_csv(meta_path, sep=sep)
        else:
            df = pd.read_csv(meta_path, sep=None, engine="python")
        df.to_csv(meta_csv, index=False)

    img_zips = _find_all_files(bundle_extract, r"\.zip$", exts={".zip"})
    img_zips = [p for p in img_zips if re.search(r"image|images|part", os.path.basename(p), re.IGNORECASE)]
    if len(img_zips) == 0:
        raise RuntimeError(f"Cannot find image zip files inside bundle: {bundle_zip}")

    for zp in sorted(img_zips):
        try:
            _unzip(zp, images_extract)
        except zipfile.BadZipFile:
            continue

    moved = _flatten_images(images_extract, images_root)
    nimg = _count_images(images_root)

    if not os.path.exists(meta_csv) or os.path.getsize(meta_csv) == 0:
        raise RuntimeError(f"Metadata missing/empty after conversion: {meta_csv}")
    if nimg == 0:
        raise RuntimeError(f"Images extraction failed, images_root is empty: {images_root}")

    print(f"[ok] metadata: {meta_csv}")
    print(f"[ok] images: {images_root}  n={nimg}  moved={moved}")
    return images_root, meta_csv

def ensure_adacbm_ham10000_json(root_dir: str) -> str:
    dest = os.path.join(root_dir, "HAM10000", "concepts", "AdaCBM_HAM10000.json")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] Exists: {dest}")
        return dest
    print(f"[wget] {ADACBM_HAM10000_JSON_RAW}")
    return wget_download(ADACBM_HAM10000_JSON_RAW, dest, resume=False)

def load_adacbm_candidate_concepts(adacbm_json_path: str) -> dict:
    d = json.load(open(adacbm_json_path, "r"))
    if not isinstance(d, dict):
        raise ValueError("Unexpected format: root is not a dict")

    per_class = {c: [] for c in HAM7}

    for k, v in d.items():
        kk = str(k).strip().lower()
        if kk not in TOPKEY_TO_HAM7:
            print(f"[warn] unknown class key in json (ignored): {k}")
            continue
        cls = TOPKEY_TO_HAM7[kk]
        if isinstance(v, list):
            per_class[cls].extend([str(x).strip() for x in v if str(x).strip()])
        elif isinstance(v, str):
            s = v.strip()
            if s:
                per_class[cls].append(s)
        elif isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, str):
                    s = vv.strip()
                    if s:
                        per_class[cls].append(s)
                elif isinstance(vv, list):
                    per_class[cls].extend([str(x).strip() for x in vv if str(x).strip()])

    for cls in HAM7:
        seen = set()
        out = []
        for s in per_class[cls]:
            if s and (s not in seen):
                out.append(s)
                seen.add(s)
        per_class[cls] = out

    missing = [c for c in HAM7 if len(per_class[c]) == 0]
    if missing:
        raise RuntimeError(f"Missing concepts for: {missing}")

    print("[concept pool sizes]")
    for cls in HAM7:
        print(f"  {cls:>5}: {len(per_class[cls])}")

    return per_class


def task_label(dx: str, task_mode="ham7") -> int:
    dx = str(dx).strip()
    if task_mode == "ham7":
        return int(HAM7_TO_ID[dx])
    if task_mode == "mel_vs_nv":
        return 1 if dx == "mel" else 0
    if task_mode == "mel_vs_rest":
        return 1 if dx == "mel" else 0
    raise ValueError("task_mode must be one of: ham7, mel_vs_nv, mel_vs_rest")


def load_ham_metadata(meta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    if "image_id" not in df.columns:
        raise ValueError("meta_csv must contain column: image_id")
    if "dx" not in df.columns:
        raise ValueError("meta_csv must contain column: dx")
    df = df[df["dx"].isin(HAM7)].copy()
    return df


def build_splits(df: pd.DataFrame, seed=42, task_mode="ham7"):
    if task_mode == "mel_vs_nv":
        df = df[df["dx"].isin(["mel", "nv"])].copy()

    if task_mode == "ham7":
        y = df["dx"].map(HAM7_TO_ID).values
    elif task_mode in ["mel_vs_nv", "mel_vs_rest"]:
        y = (df["dx"] == "mel").astype(int).values
    else:
        raise ValueError("task_mode must be one of: ham7, mel_vs_nv, mel_vs_rest")

    idx = np.arange(len(df))
    tr_idx, rest_idx = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    y_rest = y[rest_idx]
    va_idx, te_idx = train_test_split(rest_idx, test_size=0.5, random_state=seed, stratify=y_rest)

    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy(), df.iloc[te_idx].copy()


def build_image_index(images_root: str):
    idx = {}
    for dirpath, _, fns in os.walk(images_root):
        for fn in fns:
            base, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXTS:
                idx[base] = os.path.join(dirpath, fn)
    return idx


def resolve_image_path(image_id: str, images_root: str, idx_map=None) -> str:
    if idx_map is not None and image_id in idx_map:
        return idx_map[image_id]
    for ext in [".jpg", ".jpeg", ".png"]:
        cand = os.path.join(images_root, f"{image_id}{ext}")
        if os.path.exists(cand):
            return cand
    idx2 = build_image_index(images_root)
    if image_id in idx2:
        return idx2[image_id]
    raise FileNotFoundError(f"Cannot find image for image_id={image_id} under {images_root}")


def pick_device_auto():
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def make_pos_neg_prompts(concepts):
    pos = [f"a dermoscopic image of a skin lesion with {c}" for c in concepts]
    neg = [f"a dermoscopic image of a skin lesion without {c}" for c in concepts]
    return pos, neg


class ImgListDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB")


def collate_pil(batch):
    return list(batch)

def _clip_features_to_tensor(x):
    # transformers>=5: CLIPModel.get_*_features returns BaseModelOutputWithPooling
    # where pooler_output is overwritten with projected features.
    if torch.is_tensor(x):
        return x
    if hasattr(x, "pooler_output") and torch.is_tensor(x.pooler_output):
        return x.pooler_output
    if hasattr(x, "text_embeds") and torch.is_tensor(x.text_embeds):
        return x.text_embeds
    if hasattr(x, "image_embeds") and torch.is_tensor(x.image_embeds):
        return x.image_embeds
    if isinstance(x, (tuple, list)):
        if len(x) > 1 and torch.is_tensor(x[1]):
            return x[1]
        if len(x) > 0 and torch.is_tensor(x[0]):
            return x[0]
    raise TypeError(f"Unexpected CLIP features type: {type(x)}")


def build_clip_text_feats(model, tokenizer, concepts, device):
    pos_prompts, neg_prompts = make_pos_neg_prompts(concepts)
    pos_and_neg = pos_prompts + neg_prompts
    C = len(concepts)
    with torch.no_grad():
        txt_inputs = tokenizer(pos_and_neg, padding=True, return_tensors="pt").to(device)
        text_feats = _clip_features_to_tensor(model.get_text_features(**txt_inputs))
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
    return text_feats[:C], text_feats[C:]


def select_concepts_topk(
    train_paths,
    train_dx_labels,
    per_class_candidates: dict,
    topk: int,
    model_id="openai/clip-vit-base-patch32",
    batch_size=256,
    num_workers=8,
    device=None,
):
    if topk < 1:
        raise ValueError("topk must be >= 1")
    if device is None:
        device = pick_device_auto()

    union = []
    seen = set()
    for cls in HAM7:
        for c in per_class_candidates[cls]:
            if c not in seen:
                union.append(c)
                seen.add(c)
    Ccand = len(union)
    print(f"[selection] candidate union size = {Ccand}")

    union_idx = {c: i for i, c in enumerate(union)}
    cand_idx_by_class = {cls: np.array([union_idx[c] for c in per_class_candidates[cls]], dtype=np.int64) for cls in HAM7}

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    image_proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    text_pos, text_neg = build_clip_text_feats(model, tokenizer, union, device)

    sum_in = np.zeros((len(HAM7), Ccand), dtype=np.float64)
    sumsq_in = np.zeros((len(HAM7), Ccand), dtype=np.float64)
    n_in = np.zeros((len(HAM7),), dtype=np.int64)

    sum_out = np.zeros((len(HAM7), Ccand), dtype=np.float64)
    sumsq_out = np.zeros((len(HAM7), Ccand), dtype=np.float64)
    n_out = np.zeros((len(HAM7),), dtype=np.int64)

    ds = ImgListDS(train_paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_pil,
    )

    y = np.array([HAM7_TO_ID[str(dx).strip()] for dx in train_dx_labels], dtype=np.int64)

    ptr = 0
    with torch.no_grad():
        for imgs in tqdm(loader, desc="[selection] scoring train images"):
            B = len(imgs)
            yb = y[ptr:ptr + B]
            ptr += B

            pix = image_proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            img_feats = _clip_features_to_tensor(model.get_image_features(pixel_values=pix))
            img_feats = torch.nn.functional.normalize(img_feats, dim=-1)

            sim_pos = img_feats @ text_pos.T
            sim_neg = img_feats @ text_neg.T
            margin = (sim_pos - sim_neg).detach().cpu().numpy().astype(np.float64)

            for cls_id in range(len(HAM7)):
                m = (yb == cls_id)
                nin = int(m.sum())
                nout = B - nin
                if nin > 0:
                    x_in = margin[m]
                    sum_in[cls_id] += x_in.sum(axis=0)
                    sumsq_in[cls_id] += (x_in * x_in).sum(axis=0)
                    n_in[cls_id] += nin
                if nout > 0:
                    x_out = margin[~m]
                    sum_out[cls_id] += x_out.sum(axis=0)
                    sumsq_out[cls_id] += (x_out * x_out).sum(axis=0)
                    n_out[cls_id] += nout

    eps = 1e-12
    tstats = np.full((len(HAM7), Ccand), -np.inf, dtype=np.float64)

    for cls_id in range(len(HAM7)):
        n1 = n_in[cls_id]
        n2 = n_out[cls_id]
        if n1 < 2 or n2 < 2:
            continue

        mu1 = sum_in[cls_id] / max(n1, 1)
        mu2 = sum_out[cls_id] / max(n2, 1)

        var1 = (sumsq_in[cls_id] - (sum_in[cls_id] * sum_in[cls_id]) / max(n1, 1)) / max(n1 - 1, 1)
        var2 = (sumsq_out[cls_id] - (sum_out[cls_id] * sum_out[cls_id]) / max(n2, 1)) / max(n2 - 1, 1)

        denom = np.sqrt(np.maximum(var1, 0.0) / n1 + np.maximum(var2, 0.0) / n2 + eps)
        tstats[cls_id] = (mu1 - mu2) / denom

    selected_per_class = {}
    selected_union = []
    used = set()

    for cls in HAM7:
        cls_id = HAM7_TO_ID[cls]
        cand_idx = cand_idx_by_class[cls]
        cand_t = tstats[cls_id, cand_idx]
        order = np.argsort(-cand_t)

        picked = []
        for j in order:
            if len(picked) >= topk:
                break
            idx = int(cand_idx[j])
            t = tstats[cls_id, idx]
            if not np.isfinite(t) or t <= 0:
                continue
            picked.append(union[idx])

        selected_per_class[cls] = picked

        for c in picked:
            if c not in used:
                selected_union.append(c)
                used.add(c)

        print(f"[selection] {cls}: picked {len(picked)} / topk={topk}")

    print(f"[selection] final selected union concepts: C={len(selected_union)}")
    return selected_union, selected_per_class


def compute_clip_hard_binary_labels(
    image_paths,
    concepts,
    model_id="openai/clip-vit-base-patch32",
    batch_size=256,
    num_workers=8,
    device=None,
):
    if device is None:
        device = pick_device_auto()

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    image_proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    text_pos, text_neg = build_clip_text_feats(model, tokenizer, concepts, device)
    C = len(concepts)

    ds = ImgListDS(image_paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_pil,
    )

    all_flags = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="CLIP hard concept labels"):
            pix = image_proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            img_feats = _clip_features_to_tensor(model.get_image_features(pixel_values=pix))
            img_feats = torch.nn.functional.normalize(img_feats, dim=-1)

            sim_pos = img_feats @ text_pos.T
            sim_neg = img_feats @ text_neg.T
            flags = (torch.stack([sim_pos, sim_neg], dim=1).argmax(dim=1) == 0)
            all_flags.append(flags.cpu())

    out = torch.cat(all_flags, dim=0)
    if out.shape[1] != C:
        raise RuntimeError(f"Bad concept label shape: {tuple(out.shape)} expected C={C}")
    return out


def write_split_pkl(
    df_split,
    images_root,
    concepts,
    concept_flags,
    out_pkl_path,
    idx_map,
    task_mode="ham7",
):
    records = []
    C = len(concepts)

    if concept_flags.shape[0] != len(df_split) or concept_flags.shape[1] != C:
        raise RuntimeError("concept_flags shape mismatch")

    for i in tqdm(range(len(df_split)), desc=f"Writing {os.path.basename(out_pkl_path)}"):
        row = df_split.iloc[i]
        img_id = row["image_id"]
        dx = row["dx"]
        img_path = resolve_image_path(img_id, images_root, idx_map=idx_map)
        concept = concept_flags[i].to(torch.int).tolist()
        records.append(
            {
                "image_path": img_path,
                "class_label": task_label(dx, task_mode=task_mode),
                "attribute_label": concept,
            }
        )

    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(records, f)

    print(f"[saved] {out_pkl_path}  items={len(records)}  C={C}")


def setup_ham10000_dataset(
    root_dir: str,
    out_dir: str,
    topk: int = 20,
    task_mode: str = "ham7",
    seed: int = 42,
    model_id: str = "openai/clip-vit-base-patch32",
    batch_size: int = 256,
    num_workers: int = 8,
    force: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    train_pkl = os.path.join(out_dir, "train.pkl")
    val_pkl = os.path.join(out_dir, "val.pkl")
    test_pkl = os.path.join(out_dir, "test.pkl")
    concepts_txt = os.path.join(out_dir, "concepts_used.txt")
    concepts_per_class_json = os.path.join(out_dir, "concepts_selected_per_class.json")

    images_root = os.path.join(out_dir, "images")
    meta_csv = os.path.join(out_dir, "HAM10000_metadata.csv")

    if (not force) and os.path.exists(train_pkl) and os.path.exists(val_pkl) and os.path.exists(test_pkl) and os.path.exists(concepts_txt):
        print("[skip] setup already complete")
        return images_root, meta_csv

    images_root, meta_csv = ensure_ham10000_raw(out_dir)

    idx_map = build_image_index(images_root)

    adacbm_json = ensure_adacbm_ham10000_json(root_dir)
    per_class_candidates = load_adacbm_candidate_concepts(adacbm_json)

    df = load_ham_metadata(meta_csv)
    df_tr, df_va, df_te = build_splits(df, seed=seed, task_mode=task_mode)
    print(f"[split] train={len(df_tr)}  val={len(df_va)}  test={len(df_te)}")

    tr_paths = [resolve_image_path(x, images_root, idx_map=idx_map) for x in df_tr["image_id"].tolist()]
    tr_dx = df_tr["dx"].tolist()

    concepts, per_class_selected = select_concepts_topk(
        train_paths=tr_paths,
        train_dx_labels=tr_dx,
        per_class_candidates=per_class_candidates,
        topk=topk,
        model_id=model_id,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    with open(concepts_txt, "w") as f:
        for c in concepts:
            f.write(c + "\n")
    with open(concepts_per_class_json, "w") as f:
        json.dump(per_class_selected, f, indent=2)

    va_paths = [resolve_image_path(x, images_root, idx_map=idx_map) for x in df_va["image_id"].tolist()]
    te_paths = [resolve_image_path(x, images_root, idx_map=idx_map) for x in df_te["image_id"].tolist()]

    tr_flags = compute_clip_hard_binary_labels(tr_paths, concepts, model_id=model_id, batch_size=batch_size, num_workers=num_workers)
    va_flags = compute_clip_hard_binary_labels(va_paths, concepts, model_id=model_id, batch_size=batch_size, num_workers=num_workers)
    te_flags = compute_clip_hard_binary_labels(te_paths, concepts, model_id=model_id, batch_size=batch_size, num_workers=num_workers)

    write_split_pkl(df_tr, images_root, concepts, tr_flags, train_pkl, idx_map, task_mode=task_mode)
    write_split_pkl(df_va, images_root, concepts, va_flags, val_pkl, idx_map, task_mode=task_mode)
    write_split_pkl(df_te, images_root, concepts, te_flags, test_pkl, idx_map, task_mode=task_mode)

    print("[done] setup complete")
    return images_root, meta_csv


class HAM10000Dataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        s = self.data_list[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return {
            "img": img,
            "class_label": torch.tensor(int(s["class_label"]), dtype=torch.long),
            "attribute_label": torch.tensor(s["attribute_label"], dtype=torch.float32),
        }


def _clamp_num_workers(requested: int) -> int:
    """Clamp DataLoader workers to the CPU affinity of the current process.

    Slurm/cgroup environments sometimes expose only a small CPU affinity set; spawning too many
    workers can slow down or freeze the job.
    """
    try:
        requested_i = int(requested)
    except Exception:
        requested_i = 0
    if requested_i <= 0:
        return 0
    try:
        max_workers = len(os.sched_getaffinity(0))
    except Exception:
        max_workers = int(os.cpu_count() or 1)
    return int(min(requested_i, max(1, max_workers)))


def load_ham10000_data(pkl_path, batch_size, is_training, encoder: str = "resnet18", num_workers: int = 8):
    # ImageNet normalization (ResNet/ViT default)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    # CLIP normalization (CLIPVisionModel default)
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    mean, std = imagenet_mean, imagenet_std
    resol = 224
    # When using encoder='medical_vit' (CLIP vision tower), switch to CLIP normalization.
    # This keeps preprocessing aligned with the backbone's pretraining recipe.
    if str(encoder).lower() in ("medical_vit", "clip"):
        mean, std = clip_mean, clip_std

    if is_training:
        tfm = transforms.Compose(
            [
                transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        tfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    ds = HAM10000Dataset(data_list=data_list, transform=tfm)
    num_workers = _clamp_num_workers(num_workers)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


if __name__ == "__main__":
    root_dir = ""#todo change to your own
    out_dir = ""

    images_root, meta_csv = setup_ham10000_dataset(
        root_dir=root_dir,
        out_dir=out_dir,
        topk=20,
        task_mode="ham7",
        seed=42,
        model_id="openai/clip-vit-base-patch32",
        batch_size=256,
        num_workers=8,
        force=False,
    )

    train_pkl = os.path.join(out_dir, "train.pkl")
    val_pkl = os.path.join(out_dir, "val.pkl")
    test_pkl = os.path.join(out_dir, "test.pkl")

    train_loader = load_ham10000_data(train_pkl, batch_size=64, is_training=True)
    b = next(iter(train_loader))
    print("img:", b["img"].shape, b["img"].dtype)
    print("class_label:", b["class_label"].shape, b["class_label"].dtype, int(b["class_label"].min()), int(b["class_label"].max()))
    print("attribute_label:", b["attribute_label"].shape, b["attribute_label"].dtype, torch.unique(b["attribute_label"]))
