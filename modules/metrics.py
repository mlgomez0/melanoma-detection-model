import torch
from sklearn.metrics import recall_score

def calculate_recall(model, dataloader, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred_labels = torch.argmax(y_logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_labels.cpu().numpy())
    
    return recall_score(y_true, y_pred, average='weighted')