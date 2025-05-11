import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append((name, layer))
    
    if index < 0 or index >= len(conv_layers):
        raise ValueError(f"Index {index} is out of range. Model has {len(conv_layers)} convolutional layers.")
    
    return conv_layers[index][1]  # Return the layer object

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture activations
        self.target_layer.register_forward_hook(self.save_activations)

        # Hook to capture gradients using register_full_backward_hook
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def save_activations(self, module, input, output):
        self.activations = output

    def generate_cam(self, input_tensor, class_idx=None):

        # input_tensor = input_tensor.requires_grad_(True)
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))

        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Normalize CAM
        cam = np.maximum(cam, 0) # ensure the array is greater 0
        cam = cam / cam.max()
        return cam
    
    def overlay_cam(self, image, cam, alpha=0.5):
        cam = np.uint8(255 * cam)
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        cam = np.stack([cam, cam, cam], axis=-1)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        overlay = (1 - alpha) * image + alpha * cam
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

    # def overlay_cam(self, image, cam, alpha=0.5):
    #     cam = np.uint8(255 * cam)
    #     cam = np.stack([cam, cam, cam], axis=-1)  # Convert to 3-channel RGB
    #     cam = np.resize(cam, image.shape)  # Resize to match image dimensions
    #     overlay = (1 - alpha) * image + alpha * cam
    #     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    #     return overlay

    def remove_hooks(self):
        self.target_layer._forward_hooks.clear()
        self.target_layer._backward_hooks.clear()


# Function to create GradCAM by layer index
def create_gradcam_by_layer_index(model, conv_layer_idx):
    """
    Create a GradCAM instance using a specific convolutional layer index
    
    Args:
        model: PyTorch model
        conv_layer_idx (int): Index of the convolutional layer to use (0-based)
        
    Returns:
        GradCAM: Initialized GradCAM instance
    """
    target_layer = get_conv_layer_by_index(model, conv_layer_idx)
    return GradCAM(model, target_layer)


# Example usage
if __name__ == "__main__":
    from models import PneumoniaCNN
    from dataloaders import val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)
    model.eval()

    # Method 1: Select target layer directly
    # target_layer = model.model.layer4[1].conv2
    # grad_cam = GradCAM(model, target_layer)

    # Method 2: Select target layer by index
    # Print convolutional layers to see what's available
    conv_layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append((name, layer))
    print(f"Found {len(conv_layers)} convolutional layers:")
    for i, (name, _) in enumerate(conv_layers):
        print(f"  {i}: {name}")
        
    # Create GradCAM using second convolutional layer (index 1)
    grad_cam = create_gradcam_by_layer_index(model, 1)

    # Get a sample input
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    input_tensor = images[0].unsqueeze(0).to(device)

    # Generate CAM
    cam = grad_cam.generate_cam(input_tensor)

    # Visualize
    image = images[0].permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5) * 255  # Denormalize
    image = image.astype(np.uint8)

    overlay = grad_cam.overlay_cam(image, cam)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()