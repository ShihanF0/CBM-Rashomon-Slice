import os
import pickle
from collections import defaultdict as ddict
from os.path import join, exists
from typing import Dict, List
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch

def _resolve_cub_base(data_dir: str):
    p = os.path.abspath(data_dir)
    if exists(join(p, "images.txt")) and exists(join(p, "images")):
        return p
    cand = join(p, "CUB_200_2011")
    if exists(join(cand, "images.txt")) and exists(join(cand, "images")):
        return cand
    raise FileNotFoundError(
        f"Cannot find CUB_200_2011 under {data_dir}. "
        f"Expect {data_dir}/CUB_200_2011 or data_dir itself to be that folder."
    )

_CONCEPTS_IN_USE = (
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101,
    106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152,
    153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194,
    196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236,
    238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277,
    283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311
)
_KEEP_IDX = [i - 1 for i in _CONCEPTS_IN_USE]  # 0-based

def extract_data_and_save(data_dir: str, save_dir: str = None, seed: int = 42, keep_idx: List[int] = None):
    base = _resolve_cub_base(data_dir)
    out_dir = os.path.abspath(save_dir or data_dir)
    os.makedirs(out_dir, exist_ok=True)

    images_txt = join(base, "images.txt")
    labels_txt = join(base, "image_class_labels.txt")
    attrs_txt  = join(base, "attributes", "image_attribute_labels.txt")

    # 1) images
    id2rel = {}
    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, rel = line.split()
            id2rel[int(img_id)] = rel

    # 2) class labels
    id2cls = {}
    with open(labels_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, cls = line.split()
            id2cls[int(img_id)] = int(cls) - 1

    N_ATT = 312
    from collections import defaultdict as ddict
    attr_bin  = ddict(lambda: [0]*N_ATT)
    attr_cert = ddict(lambda: [0]*NATT) if False else ddict(lambda: [0]*N_ATT)
    attr_unc  = ddict(lambda: [0.0]*N_ATT)
    # 1=not visible, 2=guessing, 3=probably, 4=definitely
    uncertainty_map = {
        1: {1: 0.0, 2: 0.5, 3: 0.75, 4: 1.0},
        0: {1: 0.0, 2: 0.5, 3: 0.25, 4: 0.0},
    }
    with open(attrs_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            file_idx = int(parts[0])
            att_idx  = int(parts[1]) - 1  # 0-based
            lab      = int(parts[2])
            cert     = int(parts[3])
            attr_bin[file_idx][att_idx]  = lab
            attr_cert[file_idx][att_idx] = cert
            attr_unc[file_idx][att_idx]  = uncertainty_map[lab][cert]

    def _maybe_slice(lst):
        if keep_idx is None:
            return lst
        return [lst[k] for k in keep_idx]

    all_samples = []
    for img_id, rel in id2rel.items():
        img_path = join(base, "images", rel)
        attr_full  = attr_bin[img_id]
        cert_full  = attr_cert[img_id]
        unc_full   = attr_unc[img_id]
        all_samples.append({
            "id": img_id,
            "img_path": img_path,
            "class_label": id2cls[img_id],
            "attribute_label": _maybe_slice(attr_full),
            "attribute_certainty": _maybe_slice(cert_full), 
            "uncertain_attribute_label": _maybe_slice(unc_full),
        })

    y_all = [s["class_label"] for s in all_samples]
    train, rest = train_test_split(all_samples, test_size=0.5, random_state=seed, stratify=y_all)
    y_rest = [s["class_label"] for s in rest]
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y_rest)

    def _dump(name, data):
        with open(join(out_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(data, f)

    _dump("train", train)
    _dump("val",   val)
    _dump("test",  test)

    used = len(keep_idx) if keep_idx is not None else N_ATT
    print(f"[CUB saved to {out_dir}] train={len(train)}, val={len(val)}, test={len(test)}, n_attr={used}")

    return {"train": train, "val": val, "test": test}

def setup_CUB_dataset(data_dir: str, seed: int = 42, use_subset: bool = True):
    out_dir = os.path.abspath(data_dir)
    if exists(join(out_dir, "train.pkl")):
        print("CUB setup already complete. Found 'train.pkl'.")
        return
    keep = _KEEP_IDX if use_subset else None
    extract_data_and_save(data_dir=out_dir, save_dir=out_dir, seed=seed, keep_idx=keep)

class CUBDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        s = self.data_list[idx]
        img_path = s.get('img_path', s.get('image_path'))

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        class_label = torch.tensor(s['class_label'], dtype=torch.long)
        attr = s.get('attribute_label', None)
        if attr is None:
            attribute_label = torch.zeros(312, dtype=torch.float32)
        else:
            attribute_label = torch.tensor(attr, dtype=torch.float32)

        return {'img': image, 'class_label': class_label, 'attribute_label': attribute_label}

def load_CUB_data(pkl_path, batch_size, is_training, resol=224, num_workers=8):  
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)

    dataset = CUBDataset(data_list=data_list, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(is_training),
        drop_last=bool(is_training),
        pin_memory=True,
        num_workers=num_workers,
    )
    return loader