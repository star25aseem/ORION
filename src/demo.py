import torch
import torchvision
import torchvision.transforms as transforms

from model import BayesianCNN
from attacks import fgsm_attack
from uncertainty import mc_dropout_predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

model = BayesianCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

image, label = next(iter(testloader))
image, label = image.to(device), label.to(device)

adv_image = fgsm_attack(model, image.clone(), label)

clean_pred, clean_unc = mc_dropout_predict(model, image)
adv_pred, adv_unc = mc_dropout_predict(model, adv_image)

print(f"Clean → Prediction: {clean_pred}, Uncertainty: {clean_unc}")
print(f"Adversarial → Prediction: {adv_pred}, Uncertainty: {adv_unc}")

