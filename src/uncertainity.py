import torch
import torch.nn.functional as F

def mc_dropout_predict(model, x, runs=20):
    model.train()  # dropout ON
    preds = []

    for _ in range(runs):
        preds.append(F.softmax(model(x), dim=1))

    preds = torch.stack(preds)
    mean = preds.mean(0)
    uncertainty = preds.var(0).mean().item()
    return mean.argmax(1).item(), uncertainty

