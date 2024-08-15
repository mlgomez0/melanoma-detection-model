import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

def confusion_matrix_graph(y, y_pred, label):
  cm = confusion_matrix(y, y_pred)
  cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
  sns.heatmap(cm_df, annot=True, cmap='Blues', cbar=False, )
  plt.title(f"Confusion matrix for {label}")

def print_metrics(y_test, y_pred, label):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation metrics
    print(f"Accuracy for {label}: {accuracy}")
    print(f"Precision for {label}: {precision}")
    print(f"Recall for {label}: {recall}")
    print(f"F1-score for {label}: {f1}")
    return accuracy, precision, recall, f1