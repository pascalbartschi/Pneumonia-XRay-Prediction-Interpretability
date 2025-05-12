# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from config import config
# from pathlib import Path
#
# transform_test_val = transforms.Compose([
#     transforms.Resize((224, 224)),
#     #transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])
#
#
#
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])
#
# data_dir = config.datapath / "chest_xray"
# train_dir = data_dir / "train"
# val_dir = data_dir / "val"
# test_dir = data_dir / "test"
#
# train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# val_dataset = datasets.ImageFolder(val_dir, transform=transform_test_val)
# test_dataset = datasets.ImageFolder(test_dir, transform=transform_test_val)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config
from pathlib import Path
import os
import re

# -----------------------------
# Utility functions to extract and check patient IDs
# -----------------------------
def extract_patient_id(filename):
    if 'person' in filename:
        return filename.split('_')[0]  # 'person1'
    elif 'IM-' in filename:
        match = re.match(r'(IM-\d+)', filename)
        if match:
            return match.group(1)  # 'IM-0001'
    return filename  # fallback to filename itself if no pattern matched

def get_patient_ids(folder):
    patient_ids = set()
    for subdir, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(('jpg', 'jpeg', 'png')):
                patient_id = extract_patient_id(f)
                patient_ids.add(patient_id)
    return patient_ids

def check_patient_leakage(train_dir, val_dir, test_dir):
    train_ids = get_patient_ids(train_dir)
    val_ids = get_patient_ids(val_dir)
    test_ids = get_patient_ids(test_dir)

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    print(f"Train Patients: {len(train_ids)}")
    print(f"Validation Patients: {len(val_ids)}")
    print(f"Test Patients: {len(test_ids)}")

    if overlap_train_val:
        raise ValueError(f"Data Leakage detected: Train/Val patient overlap: {overlap_train_val}")
    if overlap_train_test:
        raise ValueError(f"Data Leakage detected: Train/Test patient overlap: {overlap_train_test}")
    if overlap_val_test:
        raise ValueError(f"Data Leakage detected: Val/Test patient overlap: {overlap_val_test}")

    print("✅ No patient-level leakage detected across splits.")

# -----------------------------
# Transforms
# -----------------------------
transform_test_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -----------------------------
# Paths
# -----------------------------
data_dir = config.datapath / "chest_xray"
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

# -----------------------------
# Check for leakage before creating loaders
# -----------------------------
check_patient_leakage(train_dir, val_dir, test_dir)

# -----------------------------
# Dataset & Loaders
# -----------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test_val)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)