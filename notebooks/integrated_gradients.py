import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur

class IntegratedGradients:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps

    def generate_attributions(self, input_tensor, target_class, baseline=None):
        """
        Compute Integrated Gradients for the input_tensor with respect to the target_class.
        """
        device = input_tensor.device
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(device)

        # Generate scaled inputs between baseline and input
        scaled_inputs = [
            baseline + (float(i) / self.steps) * (input_tensor - baseline)
            for i in range(self.steps + 1)
        ]

        grads = []
        self.model.eval()

        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_()
            output = self.model(scaled_input)
            loss = output[0, target_class]
            self.model.zero_grad()
            loss.backward()
            grads.append(scaled_input.grad.detach().clone())

        grads = torch.stack(grads)
        avg_grads = grads.mean(dim=0)
        integrated_grad = (input_tensor - baseline) * avg_grads
        return integrated_grad

    def process_attributions(self, attributions, gamma=1.5):
        """
        Normalize and optionally apply gamma correction to make IG attributions pop visually.
        """
        attr = attributions.squeeze().detach().cpu().numpy()  # [C, H, W]
        attr = np.mean(np.abs(attr), axis=0)  # [H, W]
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        attr = np.power(attr, gamma)  # apply gamma correction for brightness
        return attr

    def overlay_attributions(self, image, attributions, alpha=0.8):
        """
        Overlay a VIRIDIS heatmap of attributions on the original image.
        """
        # Resize attribution map if necessary
        if attributions.shape != image.shape[:2]:
            attributions = cv2.resize(attributions, (image.shape[1], image.shape[0]))

        heatmap = np.uint8(255 * attributions)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (1 - alpha) * image + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

# Optional functional interface
def integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    ig = IntegratedGradients(model, steps)
    return ig.generate_attributions(input_tensor, target_class, baseline)

# Example usage
if __name__ == "__main__":
    from models import PneumoniaCNN
    from dataloaders import val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("../model_state_dicts/cnn_model_transformed.pt", map_location=device))
    model.eval()

    ig = IntegratedGradients(model, steps=50)

    # Get sample image
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    input_tensor = images[0].unsqueeze(0).to(device)

    # Blurred baseline (optional)
    blur = GaussianBlur(kernel_size=11)
    baseline = blur(images[0]).unsqueeze(0).to(device)

    # Compute attributions
    attributions = ig.generate_attributions(input_tensor, target_class=1, baseline=baseline)
    attr_map = ig.process_attributions(attributions)

    # Preprocess original image to match Grad-CAM pipeline
    orig_img = images[0].permute(1, 2, 0).cpu().numpy()
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
    orig_img = (orig_img * 255).astype(np.uint8)

    # Overlay and visualize
    overlay = ig.overlay_attributions(orig_img, attr_map)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Integrated Gradients Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()