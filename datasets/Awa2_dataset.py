import os
import requests
import zipfile
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm


# The Awa2Dataset Class
class Awa2Dataset(Dataset):
    """
    Loads Awa2 data from a pre-processed list of dictionaries.
    Each dictionary in the list should contain 'image_path', 'class_label', and 'attribute_label'.
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
def load_awa2_data(pkl_path, batch_size, is_training):
    """
    Loads a specific data split and returns a DataLoader.
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
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)

    dataset = Awa2Dataset(data_list=data_list, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=4,          # speed up data loading
        pin_memory=True,        # faster host->GPU transfer
        persistent_workers=is_training and True,  # keep workers alive in training
    )
    return loader

# One-Time Setup Function
def download_file(url, target_path):
    """
    Downloads a file from a URL to a target path with a progress bar.
    """
    filename = os.path.basename(target_path)
    try:
        # Use requests to get the file stream
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            # Get the total file size from the headers
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            
            # Define the block size for downloading
            block_size = 8192

            # Create the tqdm progress bar
            print(f"Downloading {filename}...")
            with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, 
                      unit_divisor=1024, desc="  > Progress") as pbar:
                with open(target_path, 'wb') as f:
                    # Iterate over the file content in chunks
                    for chunk in r.iter_content(chunk_size=block_size):
                        # Write the chunk to the file
                        f.write(chunk)
                        # Update the progress bar
                        pbar.update(len(chunk))
        
        # Final check to see if the download was complete
        if total_size_in_bytes != 0 and os.path.getsize(target_path) != total_size_in_bytes:
            print(f"Error: Download incomplete for {filename}")
            return False

        print(f"  > Successfully downloaded {filename}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  > Successfully extracted to {extract_to}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file."); return False

def flatten_awa2(data_dir):
    inner = os.path.join(data_dir, "Animals_with_Attributes2")
    if not os.path.isdir(inner): return
    for name in os.listdir(inner):
        src = os.path.join(inner, name)
        dst = os.path.join(data_dir, name)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    try: os.rmdir(inner)
    except OSError: pass

def setup_awa2_dataset(data_dir, seed=42):
    """
    One-time setup for AwA2 with a stratified 2/1/1 split:
    train ≈ 50%, val ≈ 25%, test ≈ 25%.
    Produces train.pkl, val.pkl, test.pkl.
    """
    train_pkl_path = os.path.join(data_dir, 'train.pkl')
    if os.path.exists(train_pkl_path):
        print("Dataset setup is already complete. Found 'train.pkl'.")
        return

    print("--- Starting one-time dataset setup. This may take a while. ---")
    BASE_URL = "" #todo change to your own path
    FILES_TO_DOWNLOAD = ["AwA2-base.zip", "AwA2-features.zip", "AwA2-data.zip"]  # AwA2-data.zip ~13GB

    os.makedirs(data_dir, exist_ok=True)
    for filename in FILES_TO_DOWNLOAD:
        zip_path = os.path.join(data_dir, filename)
        if not os.path.exists(zip_path):
            download_file(BASE_URL + filename, zip_path)
        else:
            print(f"File '{filename}' already exists. Skipping download.")

    EXTRACTED_FOLDER_PATH = data_dir
    if not os.path.exists(os.path.join(EXTRACTED_FOLDER_PATH, "images")):
        print("\nExtracting files...")
        for filename in FILES_TO_DOWNLOAD:
            unzip_file(os.path.join(data_dir, filename), data_dir)
    else:
        print("\nData already seems to be extracted. Skipping unzipping.")

    flatten_awa2(data_dir)

    print("\n--- Preparing data splits (train/val/test = 2/1/1) ---")
    with open(os.path.join(EXTRACTED_FOLDER_PATH, 'classes.txt')) as f:
        id_to_name = dict([line.strip().split('\t') for line in f.readlines()])
        name_to_id = {v: int(k) for k, v in id_to_name.items()}

    with open(os.path.join(EXTRACTED_FOLDER_PATH, 'trainclasses.txt')) as f:
        trainval_ids = {name_to_id[line.strip()] for line in f.readlines()}
    with open(os.path.join(EXTRACTED_FOLDER_PATH, 'testclasses.txt')) as f:
        test_ids = {name_to_id[line.strip()] for line in f.readlines()}

    with open(os.path.join(EXTRACTED_FOLDER_PATH, "Features/ResNet101", "AwA2-filenames.txt")) as f:
        all_filenames = [line.strip() for line in f.readlines()]

    all_labels_by_sample = np.loadtxt(
        os.path.join(EXTRACTED_FOLDER_PATH, "Features/ResNet101", "AwA2-labels.txt"),
        dtype=int
    )
    attribute_matrix_by_class = np.loadtxt(
        os.path.join(EXTRACTED_FOLDER_PATH, "predicate-matrix-binary.txt"),
        dtype=np.float32
    )

    all_samples = []
    for i in tqdm(range(len(all_filenames)), desc="Pooling all samples"):
        class_label_1based = int(all_labels_by_sample[i])
        class_label_0based = class_label_1based - 1
        attribute_label = attribute_matrix_by_class[class_label_0based]
        sample_data = {
            "image_path": os.path.join(
                EXTRACTED_FOLDER_PATH, "JPEGImages",
                id_to_name[str(class_label_1based)], all_filenames[i]
            ),
            "class_label": class_label_0based,
            "attribute_label": attribute_label
        }
        all_samples.append(sample_data)

    labels_all = [s["class_label"] for s in all_samples]

    print("\nSplitting into train (50%) and rest (50%) ...")
    train_samples, rest_samples = train_test_split(
        all_samples,
        test_size=0.5,
        random_state=seed,
        stratify=labels_all
    )

    labels_rest = [s["class_label"] for s in rest_samples]
    print("Splitting the rest into val (25%) and test (25%) ...")
    val_samples, test_samples = train_test_split(
        rest_samples,
        test_size=0.5,
        random_state=seed,
        stratify=labels_rest
    )

    print("\n--- Saving metadata splits to .pkl files ---")
    with open(os.path.join(data_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(data_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(data_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    print("Done. Saved train.pkl (≈50%), val.pkl (≈25%), test.pkl (≈25%).")
