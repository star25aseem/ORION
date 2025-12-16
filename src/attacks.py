import torch
import torch.nn.functional as F

def fgsm_attack(model, image, label, epsilon=0.2):
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    perturbed = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed, 0, 1)

