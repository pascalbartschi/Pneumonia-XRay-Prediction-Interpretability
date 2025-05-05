import torch

def integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(input_tensor.device)


    # Scale inputs and compute gradients
    intermediary_images = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    grads = []

    model.eval()
    for scaled_input in intermediary_images:
        scaled_input = scaled_input.requires_grad_()
        output = model(scaled_input)
        model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        grads.append(scaled_input.grad.detach().clone())

    grads = torch.stack(grads)  # [steps+1, C, H, W]
    avg_grads = grads.mean(dim=0)
    integrated_grad = (input_tensor - baseline) * avg_grads
    return integrated_grad