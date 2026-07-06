import os
import numpy as np
import pickle
import torch
import torchvision
import shutil
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
import os, requests, hashlib

def save_cifar10_filtered(root_dir: str):
    owner = "Trustworthy-ML-Lab"
    repo  = "Label-free-CBM"
    path_in_repo = "data/concept_sets/cifar10_filtered.txt"
    raw_url = "" #todo change to your own path

    dest_dir = os.path.join(root_dir, "cifar10")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "cifar10_filtered.txt")

    if os.path.exists(dest_path):
        print(f"[skip] Exists: {dest_path}")
        return dest_path

    print(f"[download] {raw_url}")
    with requests.get(raw_url, timeout=30) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
    print(f"[saved] {dest_path}")
    return dest_path

def process_cifar10(root_dir):
    cifar = "cifar10"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"

    save_cifar10_filtered(root_dir)
    with open(f"{root_dir}/{cifar}/{cifar}_filtered.txt", "r") as f:
        concept_list = [line.strip() for line in f]
    pos_and_neg_concepts = concept_list + [f"not {c}" for c in concept_list]
    C = len(concept_list)

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    image_proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    with torch.no_grad():
        txt_inputs = tokenizer(pos_and_neg_concepts, padding=True, return_tensors="pt").to(device)
        text_feats = _clip_features_to_tensor(model.get_text_features(**txt_inputs))
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)

    def _compute_for(split_train: bool, save_path: str):
        ds = torchvision.datasets.CIFAR10(root=f"{root_dir}/{cifar}", train=split_train, download=True)
        def collate_pil(batch):
            imgs, labels = zip(*batch)
            return list(imgs), torch.tensor(labels)

        loader = DataLoader(
            ds, batch_size=512, shuffle=False,
            num_workers=8, pin_memory=True,
            collate_fn=collate_pil              
        )

        all_flags = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc=f"CLIP concepts ({'train' if split_train else 'test'})"):
                pix = image_proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
                img_feats = _clip_features_to_tensor(model.get_image_features(pixel_values=pix))
                img_feats = torch.nn.functional.normalize(img_feats, dim=-1)

                sim = img_feats @ text_feats.T   # [B, 2C]
                sim = sim.view(sim.size(0), 2, C)
                flags = (sim.argmax(dim=1) == 0) # [B, C] 
                all_flags.append(flags.cpu())

        all_flags = torch.cat(all_flags, dim=0)
        torch.save(all_flags, save_path)
        print(f"Saved: {save_path}  shape={tuple(all_flags.shape)}")


    _compute_for(True,  f"{root_dir}/{cifar}/{cifar}_train_all_concept_labels.pt")
    _compute_for(False, f"{root_dir}/{cifar}/{cifar}_test_all_concept_labels.pt")

def make_cifar10_splits(root_dir, seed=42):
    cifar = "cifar10"
    ds_train = torchvision.datasets.CIFAR10(root=f"{root_dir}/{cifar}", train=True,  download=True)
    ds_test  = torchvision.datasets.CIFAR10(root=f"{root_dir}/{cifar}", train=False, download=True)
    y_train = np.array(ds_train.targets)
    y_test  = np.array(ds_test.targets)
    y_all   = np.concatenate([y_train, y_test], axis=0)
    idx_all = np.arange(len(y_all))
    rng = np.random.RandomState(seed)

    train_idx, rest_idx = train_test_split(
        idx_all, test_size=0.5, stratify=y_all, random_state=seed
    )
    y_rest = y_all[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=y_rest, random_state=seed
    )
    return {"train": np.array(train_idx), "val": np.array(val_idx), "test": np.array(test_idx)}

def write_split_to_pkl_cifar10(root_dir: str, data_dir: str, split_name: str, indices: np.ndarray):
    cifar = "cifar10"
    os.makedirs(data_dir, exist_ok=True)

    if split_name.startswith("test"):
        save_dir = os.path.join(data_dir, "images", "test", split_name)
    else:
        save_dir = os.path.join(data_dir, "images", split_name)
    os.makedirs(save_dir, exist_ok=True)

    ds_train = torchvision.datasets.CIFAR10(root=f"{root_dir}/{cifar}", train=True,  download=True)
    ds_test  = torchvision.datasets.CIFAR10(root=f"{root_dir}/{cifar}", train=False, download=True)

    concept_train = torch.load(f"{root_dir}/{cifar}/{cifar}_train_all_concept_labels.pt")
    concept_test  = torch.load(f"{root_dir}/{cifar}/{cifar}_test_all_concept_labels.pt")

    records = []
    for gidx in tqdm(indices, desc=f"Writing {split_name}"):
        if gidx < len(ds_train):
            image, class_label = ds_train[gidx]
            concept = concept_train[gidx].to(torch.int).tolist()
        else:
            j = gidx - len(ds_train)
            image, class_label = ds_test[j]
            concept = concept_test[j].to(torch.int).tolist()

        img_filename = f"{gidx:06d}.png"
        img_path = os.path.join(save_dir, img_filename)
        image.save(img_path)

        records.append({
            "image_path": img_path,
            "class_label": int(class_label),
            "attribute_label": concept
        })

    out_pkl = os.path.join(data_dir, f"{split_name}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {split_name}: {out_pkl}  (items={len(records)})")


def setup_cifar10_dataset(root_dir, data_dir):
    train_pkl_path = os.path.join(data_dir, "train.pkl")
    if os.path.exists(train_pkl_path):
        print("Dataset setup is already complete. Found 'train.pkl'.")
        return

    process_cifar10(root_dir)

    splits = make_cifar10_splits(root_dir)

    for split_name, idx in splits.items():
        write_split_to_pkl_cifar10(root_dir, data_dir, split_name, np.array(idx))

    print("\nAll conversion tasks are complete!")

class Cifar10Dataset(Dataset):
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
            "class_label": torch.tensor(s["class_label"], dtype=torch.long),
            "attribute_label": torch.tensor(s["attribute_label"], dtype=torch.float32),
        }

def load_cifar10_data(pkl_path, batch_size, is_training):
    resol = 224
    if is_training:
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)
    ds = Cifar10Dataset(data_list, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=is_training, drop_last=is_training)
def _clip_features_to_tensor(x):
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
