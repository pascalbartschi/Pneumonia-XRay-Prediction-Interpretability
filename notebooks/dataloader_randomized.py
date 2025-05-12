from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config
import numpy as np



transform_test_val = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Grayscale(num_output_channels=1),
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


data_dir = config.datapath / "chest_xray"
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"


train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test_val)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test_val)

# # Randomize labels
# np.random.seed(42)
# randomized_labels = np.random.permutation(train_dataset.targets)
#
# # Overwrite the labels in the dataset
# train_dataset.targets = randomized_labels.tolist()



num_samples = len(train_dataset)


np.random.seed(42)
new_labels = np.array([0] * (num_samples // 2) + [1] * (num_samples - num_samples // 2))
np.random.shuffle(new_labels)


train_dataset.targets = new_labels.tolist()

# verify the balance
unique, counts = np.unique(train_dataset.targets, return_counts=True)
print(f"New class distribution: {dict(zip(unique, counts))}")



randomized_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
