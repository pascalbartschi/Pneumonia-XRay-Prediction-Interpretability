# Import necessary modules
# from models import PneumoniaCNN, PneumoniaResNet
from grad_cam import GradCAM
from integrated_gradients import integrated_gradients, IntegratedGradients
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import GaussianBlur
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################## HELPER FUNCTIONS ########################

# Define get_samples_by_label function
def get_samples_by_label(loader, label, n=5):
    """
    Get n samples of a specific label from the data loader
    
    Args:
        loader: Data loader
        label (int): Label to filter by
        n (int): Number of samples to get
        
    Returns:
        samples: List of samples with the specified label
    """
    samples = []
    
    for images, labels in loader:
        for i in range(len(labels)):
            if labels[i] == label and len(samples) < n:
                samples.append(images[i])
            if len(samples) >= n:
                break
        if len(samples) >= n:
            break
    
    return samples

def get_conv_layer_by_index(model, index):
    """
    Get the nth convolutional layer from a model
    
    Args:
        model: PyTorch model
        index (int): Index of the convolutional layer (0-based)
        
    Returns:
        layer: The selected convolutional layer
    """
    conv_layers = []
    
    # Collect all conv layers in the model
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layers.append((name, layer))
    
    if index < 0 or index >= len(conv_layers):
        raise ValueError(f"Index {index} is out of range. Model has {len(conv_layers)} convolutional layers.")
    
    return conv_layers[index][1]  # Return the layer object

######################## ViSUALIZATION FUNCTIONS ########################


def get_baseline(image, baseline_type='blur'):
    if baseline_type == 'gray':
        return torch.ones_like(image) * 0.5
    elif baseline_type == 'noise':
        return torch.randn_like(image) * 0.01  # Small noise baseline
    elif baseline_type == 'zero':
        return torch.zeros_like(image)
    elif baseline_type == 'blur':
        return GaussianBlur(kernel_size=21)(image)
    else:
        raise ValueError(f"Unknown baseline_type '{baseline_type}'")


# Visualize Integrated Gradients for second conv layer
def visualize_ig_for_images(conv_layer_idx, models, loader, n=5, model_names=None, baseline_type='blur'):
    """
    Visualize Integrated Gradients for a specific convolutional layer for normal and pneumonia images.

    Args:
        conv_layer_idx (int): Index of the convolutional layer to use
        models (list): List of models to compare
        loader: Data loader to get samples from
        n (int): Number of samples to get from each class
        model_names (list, optional): List of names for the models for plot titles. Default is ["Model 1", "Model 2"]
        baseline_type (str): Type of baseline to use ('blur', 'gray', 'noise', 'zero')
    """
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(models))]

    images_normal = get_samples_by_label(loader, 0, n=n)
    images_pneumonia = get_samples_by_label(loader, 1, n=n)

    target_layers = [get_conv_layer_by_index(model, conv_layer_idx) for model in models]
    igs = [IntegratedGradients(model, steps=50) for model in models]

    fig, axes = plt.subplots(2, n, figsize=(20, 10))
    subplot_labels = [chr(65 + i) for i in range(2 * n)]

    # Normal images
    for i, image_normal in enumerate(images_normal):
        input_tensor_normal = image_normal.unsqueeze(0).to(device)
        baseline_normal = get_baseline(image_normal, baseline_type).unsqueeze(0).to(device)

        attr_normal = [ig.generate_attributions(input_tensor_normal, target_class=0, baseline=baseline_normal) for ig in
                       igs]
        attr_map_normal = [ig.process_attributions(attr) for ig, attr in zip(igs, attr_normal)]

        orig_normal = image_normal.permute(1, 2, 0).numpy()
        orig_normal = (orig_normal - orig_normal.min()) / (orig_normal.max() - orig_normal.min())
        orig_normal = (orig_normal * 255).astype(np.uint8)

        overlay_ig_normal = [ig.overlay_attributions(orig_normal.copy(), attr_map) for ig, attr_map in
                             zip(igs, attr_map_normal)]

        axes[0, i].imshow(overlay_ig_normal[0])
        axes[0, i].axis("off")
        axes[0, i].text(0.05, 0.05, subplot_labels[i], transform=axes[0, i].transAxes,
                        fontsize=14, fontweight='bold', color='white',
                        bbox=dict(facecolor='black', alpha=0.8))

    # Pneumonia images
    for i, image_pneumonia in enumerate(images_pneumonia):
        input_tensor_pneumonia = image_pneumonia.unsqueeze(0).to(device)
        baseline_pneumonia = get_baseline(image_pneumonia, baseline_type).unsqueeze(0).to(device)

        attr_pneumonia = [ig.generate_attributions(input_tensor_pneumonia, target_class=1, baseline=baseline_pneumonia)
                          for ig in igs]
        attr_map_pneumonia = [ig.process_attributions(attr) for ig, attr in zip(igs, attr_pneumonia)]

        orig_pneumonia = image_pneumonia.permute(1, 2, 0).numpy()
        orig_pneumonia = (orig_pneumonia - orig_pneumonia.min()) / (orig_pneumonia.max() - orig_pneumonia.min())
        orig_pneumonia = (orig_pneumonia * 255).astype(np.uint8)

        overlay_ig_pneumonia = [ig.overlay_attributions(orig_pneumonia.copy(), attr_map) for ig, attr_map in
                                zip(igs, attr_map_pneumonia)]

        axes[1, i].imshow(overlay_ig_pneumonia[0])
        axes[1, i].axis("off")
        axes[1, i].text(0.05, 0.05, subplot_labels[n + i], transform=axes[1, i].transAxes,
                        fontsize=14, fontweight='bold', color='white',
                        bbox=dict(facecolor='black', alpha=0.8))

    plt.tight_layout()
    plt.show()

def visualize_gradcam_for_images(conv_layer_idx, models, loader, n=5, model_names=None):
    """
    Visualize Grad-CAM for a specific convolutional layer for normal and pneumonia images.
    
    Args:
        conv_layer_idx (int): Index of the convolutional layer to use
        models (list): List of models to compare
        loader: Data loader to get samples from
        n (int): Number of samples to get from each class
        model_names (list, optional): List of names for the models for plot titles. Default is ["Model 1", "Model 2"]
    """
    # Set default model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    # Get samples
    images_normal = get_samples_by_label(loader, 0, n=n)
    images_pneumonia = get_samples_by_label(loader, 1, n=n)
    
    # Get appropriate target layers by index
    target_layers = [get_conv_layer_by_index(model, conv_layer_idx) for model in models]

    # Initialize Grad-CAM for all models
    grad_cams = [GradCAM(model, layer) for model, layer in zip(models, target_layers)]
    
    # Create figure
    fig, axes = plt.subplots(2, n, figsize=(20, 10))
    
    # Define subplot labels starting at A
    subplot_labels = [chr(65 + i) for i in range(2 * n)]
    
    # Visualize normal images
    for i, image_normal in enumerate(images_normal):
        input_tensor_normal = image_normal.unsqueeze(0).to(device)
        
        # Generate CAMs
        cam_normal = [gc.generate_cam(input_tensor_normal) for gc in grad_cams]
        
        # Process original image for display
        orig_normal = image_normal.permute(1, 2, 0).numpy()
        orig_normal = (orig_normal - orig_normal.min()) / (orig_normal.max() - orig_normal.min())
        orig_normal = (orig_normal * 255).astype(np.uint8)
        
        # Overlay CAMs
        overlay_cam_normal = [gc.overlay_cam(orig_normal.copy(), cam) for gc, cam in zip(grad_cams, cam_normal)]
        
        # Display the first model's CAM
        axes[0, i].imshow(overlay_cam_normal[0])
        # axes[0, i].set_title(f"Normal {i+1}")
        axes[0, i].axis("off")
        axes[0, i].text(0.05, 0.05, subplot_labels[i], transform=axes[0, i].transAxes, 
                        fontsize=14, fontweight='bold', color='white', 
                        bbox=dict(facecolor='black', alpha=0.8))
    
    # Visualize pneumonia images
    for i, image_pneumonia in enumerate(images_pneumonia):
        input_tensor_pneumonia = image_pneumonia.unsqueeze(0).to(device)
        
        # Generate CAMs
        cam_pneumonia = [gc.generate_cam(input_tensor_pneumonia) for gc in grad_cams]
        
        # Process original image for display
        orig_pneumonia = image_pneumonia.permute(1, 2, 0).numpy()
        orig_pneumonia = (orig_pneumonia - orig_pneumonia.min()) / (orig_pneumonia.max() - orig_pneumonia.min())
        orig_pneumonia = (orig_pneumonia * 255).astype(np.uint8)
        
        # Overlay CAMs
        overlay_cam_pneumonia = [gc.overlay_cam(orig_pneumonia.copy(), cam) for gc, cam in zip(grad_cams, cam_pneumonia)]
        
        # Display the first model's CAM
        axes[1, i].imshow(overlay_cam_pneumonia[0])
        # axes[1, i].set_title(f"Pneumonia {i+1}")
        axes[1, i].axis("off")
        axes[1, i].text(0.05, 0.05, subplot_labels[n + i], transform=axes[1, i].transAxes, 
                        fontsize=14, fontweight='bold', color='white', 
                        bbox=dict(facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_comparisons(idx, conv_layer_idx, models, loader, n=5, model_names=None, baseline_type='gray'):
    """
    Visualize comparison between models using Grad-CAM and Integrated Gradients
    
    Args:
        idx (int): Index of the sample to visualize
        conv_layer_idx (int): Index of the convolutional layer to use
        models (list): List of models to compare
        loader: Data loader to get samples from
        n (int): Number of samples to get from each class
        model_names (list, optional): List of names for the models for plot titles. Default is ["Model 1", "Model 2"]
    """
    # Set default model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    # Ensure we have enough model names
    if len(model_names) < len(models):
        model_names.extend([f"Model {i+1}" for i in range(len(model_names), len(models))])
    
    # Get samples
    images_normal = get_samples_by_label(loader, 0, n=n)
    images_pneumonia = get_samples_by_label(loader, 1, n=n)

    
    # Get appropriate target layers by index
    target_layers = [get_conv_layer_by_index(model, conv_layer_idx) for model in models]

    # Initialize Grad-CAMs for all models
    grad_cams = [GradCAM(model, layer) for model, layer in zip(models, target_layers)]
    
    # Initialize Integrated Gradients for all models
    igs = [IntegratedGradients(model, steps=50) for model in models]
    
    # Select images
    image_normal = images_normal[idx]
    image_pneumonia = images_pneumonia[idx]
    
    # Prepare input tensors
    input_tensor_normal = image_normal.unsqueeze(0).to(device)
    input_tensor_pneumonia = image_pneumonia.unsqueeze(0).to(device)
    
    # Create blurred baselines
    #blur_transform = GaussianBlur(kernel_size=11)
    #baseline_normal = blur_transform(image_normal).unsqueeze(0).to(device)
    #baseline_pneumonia = blur_transform(image_pneumonia).unsqueeze(0).to(device)

    baseline_normal = get_baseline(image_normal, baseline_type).unsqueeze(0).to(device)
    baseline_pneumonia = get_baseline(image_pneumonia, baseline_type).unsqueeze(0).to(device)

    
    # Generate attributions for Integrated Gradients
    attr_normal = [ig.generate_attributions(input_tensor_normal, target_class=0, baseline=baseline_normal) for ig in igs]
    attr_pneumonia = [ig.generate_attributions(input_tensor_pneumonia, target_class=1, baseline=baseline_pneumonia) for ig in igs]
    
    # Process attributions
    attr_map_normal = [ig.process_attributions(attr) for ig, attr in zip(igs, attr_normal)]
    attr_map_pneumonia = [ig.process_attributions(attr) for ig, attr in zip(igs, attr_pneumonia)]
    
    # Generate CAMs
    cam_normal = [gc.generate_cam(input_tensor_normal) for gc in grad_cams]
    cam_pneumonia = [gc.generate_cam(input_tensor_pneumonia) for gc in grad_cams]
    
    # Process original images for display
    orig_normal = image_normal.permute(1, 2, 0).numpy()
    orig_normal = (orig_normal - orig_normal.min()) / (orig_normal.max() - orig_normal.min())
    orig_normal = (orig_normal * 255).astype(np.uint8)
    
    orig_pneumonia = image_pneumonia.permute(1, 2, 0).numpy()
    orig_pneumonia = (orig_pneumonia - orig_pneumonia.min()) / (orig_pneumonia.max() - orig_pneumonia.min())
    orig_pneumonia = (orig_pneumonia * 255).astype(np.uint8)
    
    # Create overlays
    overlay_ig_normal = [ig.overlay_attributions(orig_normal.copy(), attr_map) for ig, attr_map in zip(igs, attr_map_normal)]
    overlay_ig_pneumonia = [ig.overlay_attributions(orig_pneumonia.copy(), attr_map) for ig, attr_map in zip(igs, attr_map_pneumonia)]
    
    overlay_cam_normal = [gc.overlay_cam(orig_normal.copy(), cam) for gc, cam in zip(grad_cams, cam_normal)]
    overlay_cam_pneumonia = [gc.overlay_cam(orig_pneumonia.copy(), cam) for gc, cam in zip(grad_cams, cam_pneumonia)]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    
    # Define subplot labels A-J for publication-ready figures
    # Top row: A-E, Bottom row: F-J
    subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # Original images (first column)
    axes[0, 0].imshow(orig_normal)
    axes[0, 0].set_title("Original (Normal)")
    axes[0, 0].axis("off")
    axes[0, 0].text(0.05, 0.05, subplot_labels[0], transform=axes[0, 0].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[1, 0].imshow(orig_pneumonia)
    axes[1, 0].set_title("Original (Pneumonia)")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.05, 0.05, subplot_labels[5], transform=axes[1, 0].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    # Model 1 (second and third columns)
    axes[0, 1].imshow(overlay_ig_normal[0])
    axes[0, 1].set_title(f"IG {model_names[0]} (Normal)")
    axes[0, 1].axis("off")
    axes[0, 1].text(0.05, 0.05, subplot_labels[1], transform=axes[0, 1].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[0, 2].imshow(overlay_cam_normal[0])
    axes[0, 2].set_title(f"Grad-CAM {model_names[0]} (Normal)")
    axes[0, 2].axis("off")
    axes[0, 2].text(0.05, 0.05, subplot_labels[2], transform=axes[0, 2].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[1, 1].imshow(overlay_ig_pneumonia[0])
    axes[1, 1].set_title(f"IG {model_names[0]} (Pneumonia)")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.05, subplot_labels[6], transform=axes[1, 1].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[1, 2].imshow(overlay_cam_pneumonia[0])
    axes[1, 2].set_title(f"Grad-CAM {model_names[0]} (Pneumonia)")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.05, subplot_labels[7], transform=axes[1, 2].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    # Model 2 (fourth and fifth columns)
    axes[0, 3].imshow(overlay_ig_normal[1])
    axes[0, 3].set_title(f"IG {model_names[1]} (Normal)")
    axes[0, 3].axis("off")
    axes[0, 3].text(0.05, 0.05, subplot_labels[3], transform=axes[0, 3].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[0, 4].imshow(overlay_cam_normal[1])
    axes[0, 4].set_title(f"Grad-CAM {model_names[1]} (Normal)")
    axes[0, 4].axis("off")
    axes[0, 4].text(0.05, 0.05, subplot_labels[4], transform=axes[0, 4].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[1, 3].imshow(overlay_ig_pneumonia[1])
    axes[1, 3].set_title(f"IG {model_names[1]} (Pneumonia)")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.05, 0.05, subplot_labels[8], transform=axes[1, 3].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    axes[1, 4].imshow(overlay_cam_pneumonia[1])
    axes[1, 4].set_title(f"Grad-CAM {model_names[1]} (Pneumonia)")
    axes[1, 4].axis("off")
    axes[1, 4].text(0.05, 0.05, subplot_labels[9], transform=axes[1, 4].transAxes, 
                    fontsize=14, fontweight='bold', color='white', 
                    bbox=dict(facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'models': models,
        'target_layers': target_layers,
        'conv_layer_idx': conv_layer_idx,
        'model_names': model_names
    }