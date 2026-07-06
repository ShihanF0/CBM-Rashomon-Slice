import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import pickle
from tqdm.auto import tqdm
from torchvision.datasets import CelebA as TVCelebA

# The CelebA Dataset Class
class CelebADataset(Dataset):
    """
    Loads CelebA data from a pre-processed list of dictionaries.
    Each dictionary should contain 'image_path', 'class_label', and 'attribute_label'.
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_info = self.data_list[idx]
        img_path = sample_info['image_path']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        return {
            'img': image_tensor,
            'class_label': torch.tensor(sample_info['class_label'], dtype=torch.long),
            'attribute_label': torch.tensor(sample_info['attribute_label'], dtype=torch.float32)
        }

# Main Data Loading Function
def load_celeba_data(pkl_path, batch_size, is_training):
    """
    Loads a specific data split and returns a DataLoader for CelebA.
    """
    resol = 224 # Image resolution for ResNet-18
    
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)), 
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: # For validation and testing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)

    dataset = CelebADataset(data_list=data_list, transform=transform)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_training,
        drop_last=is_training
    )
    return loader

def binary_vector_to_decimal(vector):
    """Helper function to convert a binary vector to a decimal integer."""
    # Ensure all elements are integers for the join
    return int("".join(map(str, map(int, vector))), 2)

def download_celeba(data_dir: str) -> None:
    base = os.path.join(data_dir, "celeba")
    attr_txt = os.path.join(base, "list_attr_celeba.txt")
    img_dir  = os.path.join(base, "img_align_celeba")

    if os.path.isfile(attr_txt) and os.path.isdir(img_dir):
        return

    os.makedirs(base, exist_ok=True)
    _ = TVCelebA(root=data_dir, split="train", download=True)
    
def setup_celeba_dataset(data_dir, seed: int = 42):
    """
    One-time setup for CelebA with a stratified 2/1/1 split:
    train ≈ 50%, val ≈ 25%, test ≈ 25%.
    Produces train.pkl, val.pkl, test.pkl.
    """
    train_pkl_path = os.path.join(data_dir, 'train.pkl')
    if os.path.exists(train_pkl_path):
        print("CelebA dataset setup is already complete.")
        return

    download_celeba(data_dir)

    print("--- Starting one-time setup for the CelebA task ---")

    celeba_base_dir = os.path.join(data_dir, 'celeba')
    attr_path = os.path.join(celeba_base_dir, "list_attr_celeba.txt")
    img_base_dir = os.path.join(celeba_base_dir, "img_align_celeba")

    if not os.path.exists(attr_path):
        print(f"Error: Attribute file not found at the new path: {attr_path}")
        print("Please ensure your 'list_attr_celeba.txt' file is located there.")
        return
    if not os.path.exists(img_base_dir):
        print(f"Error: Image directory not found at the new path: {img_base_dir}")
        print("Please ensure your 'img_align_celeba' folder is located there.")
        return

    print("Loading attributes and finding the 8 most balanced...")
    df_attr = pd.read_csv(attr_path, delim_whitespace=True, header=1)
    df_attr_binary = (df_attr + 1) // 2  
    balance_scores = (df_attr_binary.mean() - 0.5).abs()
    balanced_attr_names = balance_scores.nsmallest(8).index.tolist()
    print(f"Selected attributes: {balanced_attr_names}")

    print("\nGenerating new class labels and concepts for each image...")
    df_selected_attrs = df_attr_binary[balanced_attr_names]

    all_samples = []
    for img_filename, row in tqdm(df_selected_attrs.iterrows(),
                                  total=len(df_selected_attrs),
                                  desc="Processing images"):
        attr_vector = row.values.tolist()
        class_label = binary_vector_to_decimal(attr_vector)
        concept_label = attr_vector[:6]

        all_samples.append({
            'image_path': os.path.join(img_base_dir, img_filename),
            'class_label': class_label,
            'attribute_label': concept_label
        })

    labels_all = [s['class_label'] for s in all_samples]

    print("\nSplitting into train (50%) and rest (50%) ...")
    try:
        train_samples, rest_samples = train_test_split(
            all_samples, test_size=0.5, random_state=seed, stratify=labels_all
        )
    except ValueError as e:
        print(f"Warning: 50/50 stratified split failed ({e}). Falling back to non-stratified split.")
        train_samples, rest_samples = train_test_split(
            all_samples, test_size=0.5, random_state=seed
        )

    print("Splitting the rest into val (25%) and test (25%) ...")
    labels_rest = [s['class_label'] for s in rest_samples]
    try:
        val_samples, test_samples = train_test_split(
            rest_samples, test_size=0.5, random_state=seed, stratify=labels_rest
        )
    except ValueError as e:
        print(f"Warning: val/test stratified split failed ({e}). Falling back to non-stratified split.")
        val_samples, test_samples = train_test_split(
            rest_samples, test_size=0.5, random_state=seed
        )

    print("\nSaving processed data to .pkl files ...")
    with open(os.path.join(data_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(data_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(data_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_samples, f)

    print("Done. Saved train.pkl (≈50%), val.pkl (≈25%), test.pkl (≈25%).")
    print("\nSplit sizes:")
    print("train:", len(train_samples))
    print("val:  ", len(val_samples))
    print("test: ", len(test_samples))


if __name__ == "__main__":
    data_dir = ''#change to your own
    
    # This single function call handles the entire download and setup workflow.
    setup_celeba_dataset(data_dir)

    # Once setup is complete, you can use the load_celeba_data function.
    print("\n--- Using the data loader for CelebA ---")
    batch_size = 64

    train_data_path = os.path.join(data_dir, 'train.pkl')
    val_data_path = os.path.join(data_dir, 'valid.pkl')
    test_data_path = os.path.join(data_dir, 'test.pkl')

    # Use the load_celeba_data function
    train_loader = load_celeba_data(train_data_path, batch_size=batch_size, is_training=True)
    val_loader = load_celeba_data(val_data_path, batch_size=batch_size, is_training=False)
    test_loader = load_celeba_data(test_data_path, batch_size=batch_size, is_training=False)
    
    print("Successfully created train_loader, val_loader, and test_loader.")

    # Verify by fetching one batch from the train_loader
    print("\nVerifying a batch from the training dataloader...")
    first_batch = next(iter(train_loader))
    
    print(f"Shape of image tensor batch: {first_batch['img'].shape}")
    print(f"Shape of class label batch: {first_batch['class_label'].shape}")
    print(f"Shape of attribute label batch: {first_batch['attribute_label'].shape}")
    print("Workflow demonstration complete.")