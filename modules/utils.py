"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import List, Tuple

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    


def predict_dataloader(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """Predicts labels and returns both predictions and true labels."""
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            pred_labels = torch.round(preds).long()

            # Flatten the predictions and labels and convert to lists
            all_preds.extend(pred_labels.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    return all_preds, all_labels
