import torch
import numpy as np
import cv2

class IntegratedGradients:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps
        
    def generate_attributions(self, input_tensor, target_class, baseline=None):
        """
        Generate attributions using Integrated Gradients method
        
        Args:
            input_tensor (torch.Tensor): Input image tensor
            target_class (int): Target class index
            baseline (torch.Tensor, optional): Baseline tensor. If None, zero tensor is used
            
        Returns:
            torch.Tensor: Attribution map
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(input_tensor.device)
        
        # Scale inputs and compute gradients
        intermediary_images = [baseline + (float(i) / self.steps) * (input_tensor - baseline) 
                               for i in range(self.steps + 1)]
        grads = []

        self.model.eval()
        for scaled_input in intermediary_images:
            scaled_input = scaled_input.requires_grad_()
            output = self.model(scaled_input)
            self.model.zero_grad()
            loss = output[0, target_class]
            loss.backward()
            grads.append(scaled_input.grad.detach().clone())

        grads = torch.stack(grads)  # [steps+1, C, H, W]
        avg_grads = grads.mean(dim=0)
        integrated_grad = (input_tensor - baseline) * avg_grads
        return integrated_grad
    
    def process_attributions(self, attributions):
        """
        Process attribution tensor into a normalized numpy array
        
        Args:
            attributions (torch.Tensor): Attribution tensor
            
        Returns:
            numpy.ndarray: Normalized attribution map
        """
        # Take absolute value and average across channels
        attr = attributions.squeeze().cpu().numpy()
        attr = np.mean(np.abs(attr), axis=0)
        
        # Normalize to [0, 1]
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr
    
    def overlay_attributions(self, image, attributions, alpha=0.5):
        """
        Overlay attribution map on the original image
        
        Args:
            image (numpy.ndarray): Original image
            attributions (numpy.ndarray): Attribution map
            alpha (float): Transparency factor
            
        Returns:
            numpy.ndarray: Overlay image
        """
        # Resize attributions to match image dimensions if needed
        if attributions.shape != image.shape[:2]:
            attributions = cv2.resize(attributions, (image.shape[1], image.shape[0]))
        
        # Create heatmap
        heatmap = np.uint8(255 * attributions)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = (1 - alpha) * image + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay


# For backward compatibility
def integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    """Legacy function for backward compatibility"""
    ig = IntegratedGradients(model, steps)
    return ig.generate_attributions(input_tensor, target_class, baseline)


# Example usage
if __name__ == "__main__":
    from models import PneumoniaCNN
    from dataloaders import val_loader
    import matplotlib.pyplot as plt
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)
    model.eval()
    
    # Initialize IntegratedGradients
    ig = IntegratedGradients(model, steps=50)
    
    # Get a sample input
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    input_tensor = images[0].unsqueeze(0).to(device)
    
    # Create blurred baseline
    from torchvision.transforms import GaussianBlur
    blur_transform = GaussianBlur(kernel_size=11)
    baseline = blur_transform(images[0]).unsqueeze(0).to(device)
    
    # Generate attributions
    attributions = ig.generate_attributions(input_tensor, target_class=0, baseline=baseline)
    attr_map = ig.process_attributions(attributions)
    
    # Process original image for display
    orig_img = images[0].permute(1, 2, 0).numpy()
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    orig_img = (orig_img * 255).astype(np.uint8)
    
    # Create overlay
    overlay = ig.overlay_attributions(orig_img, attr_map)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_img)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Integrated Gradients")
    plt.imshow(overlay)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()