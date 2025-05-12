import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import GaussianBlur
from PIL import Image

from models import PneumoniaCNN
from dataloaders import test_loader
from integrated_gradients import integrated_gradients

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = PneumoniaCNN().to(device)
#model = torch.load("cnn_model.pt", map_location=device)
model.load_state_dict(torch.load("../model_state_dicts/cnn_model_transformed.pt", map_location=device))
model.eval()

# Collect 5 healthy and 5 pneumonia images
samples = []
labels = []

for inputs, targets in test_loader:
    for x, y in zip(inputs, targets):
        if len(samples) >= 10:
            break
        if (y.item() == 0 and labels.count(0) < 5) or (y.item() == 1 and labels.count(1) < 5):
            samples.append(x)
            labels.append(y.item())
    if len(samples) >= 10:
        break

# Define blur transform
blur_transform = GaussianBlur(kernel_size=11)

# Run IG on each sample
for i, (input_img, label) in enumerate(zip(samples, labels)):
    input_tensor = input_img.unsqueeze(0).to(device)

    # Create blurred baseline
    baseline = blur_transform(input_img).unsqueeze(0).to(device)

    # Compute Integrated Gradients with blurred baseline
    attributions = integrated_gradients(model, input_tensor, target_class=label, baseline=baseline)

    # Process attribution map
    attr = attributions.squeeze().cpu().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    attr = (attr - attr.min()) / (attr.max() - attr.min())

    # Process original image for display
    orig_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())

    # Plot
    plt.figure(figsize=(8, 4))
    plt.suptitle(f"Sample {i + 1} | Label: {'Healthy' if label == 0 else 'Pneumonia'}", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_img)

    plt.subplot(1, 2, 2)
    plt.title("Integrated Gradients (Blurred Baseline)")
    plt.imshow(attr, cmap='viridis')

    plt.tight_layout()
    plt.show()