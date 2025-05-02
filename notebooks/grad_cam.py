import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook to capture gradients
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):
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
        activations = self.target_layer.output.cpu().data.numpy()[0]

        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))

        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        return cam

    def overlay_cam(self, image, cam, alpha=0.5):
        cam = np.uint8(255 * cam)
        cam = np.stack([cam, cam, cam], axis=-1)  # Convert to 3-channel RGB
        cam = np.resize(cam, image.shape)  # Resize to match image dimensions
        overlay = (1 - alpha) * image + alpha * cam
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

# Example usage
if __name__ == "__main__":
    from models import PneumoniaCNN
    from dataloaders import val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaResNet().to(device)
    model.eval()

    # Select a target layer (e.g., the last convolutional layer)
    target_layer = model.model.layer4[1].conv2

    grad_cam = GradCAM(model, target_layer)

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