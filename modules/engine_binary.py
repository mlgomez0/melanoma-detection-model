import torch
import torchmetrics
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

import torch
import torchmetrics
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float, float]:
    """Function to train a PyTorch model and calculate loss, accuracy, recall, and F1-score."""
    
    model.train()

    # Initialize metrics
    train_loss, train_acc, train_recall, train_f1 = 0, 0, 0, 0
    recall = torchmetrics.Recall(task="binary").to(device)
    f1 = torchmetrics.F1Score(task="binary").to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).float()  # Convert targets to float

        y_pred = model(X)

        loss = loss_fn(y_pred, y.unsqueeze(1))  # Adjust target shape to match prediction shape
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.round(y_pred)  # For binary classification, round the sigmoid output to get the class
        train_acc += (y_pred_class == y.unsqueeze(1)).sum().item() / len(y_pred)
        
        # Calculate recall and F1
        train_recall += recall(y_pred_class, y.unsqueeze(1)).item()
        train_f1 += f1(y_pred_class, y.unsqueeze(1)).item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_recall /= len(dataloader)
    train_f1 /= len(dataloader)
    
    return train_loss, train_acc, train_recall, train_f1

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float, float]:
    """Function to test a PyTorch model and calculate loss, accuracy, recall, and F1-score."""

    model.eval() 
    test_loss, test_acc, test_recall, test_f1 = 0, 0, 0, 0
    recall = torchmetrics.Recall(task="binary").to(device)
    f1 = torchmetrics.F1Score(task="binary").to(device)

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device).float()  # Convert targets to float

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y.unsqueeze(1))  # Adjust target shape to match prediction shape
            test_loss += loss.item()

            test_pred_labels = torch.round(test_pred_logits)  # For binary classification, round the sigmoid output to get the class
            test_acc += (test_pred_labels == y.unsqueeze(1)).sum().item() / len(test_pred_labels)
            
            # Calculate recall and F1
            test_recall += recall(test_pred_labels, y.unsqueeze(1)).item()
            test_f1 += f1(test_pred_labels, y.unsqueeze(1)).item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_recall /= len(dataloader)
    test_f1 /= len(dataloader)
    
    return test_loss, test_acc, test_recall, test_f1

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer:torch.utils.tensorboard.writer.SummaryWriter=None,
          example_input=None) -> Dict[str, List]:
    """Trains and tests a PyTorch model, including loss, accuracy, recall, and F1-score."""

    results = {"train_loss": [],
               "train_acc": [],
               "train_recall": [],
               "train_f1": [],
               "test_loss": [],
               "test_acc": [],
               "test_recall": [],
               "test_f1": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_recall, train_f1 = train_step(model=model,
                                                                   dataloader=train_dataloader,
                                                                   loss_fn=loss_fn,
                                                                   optimizer=optimizer,
                                                                   device=device)
        test_loss, test_acc, test_recall, test_f1 = test_step(model=model,
                                                              dataloader=test_dataloader,
                                                              loss_fn=loss_fn,
                                                              device=device)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"train_recall: {train_recall:.4f} | "
          f"train_f1: {train_f1:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"test_recall: {test_recall:.4f} | "
          f"test_f1: {test_f1:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_recall"].append(train_recall)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_recall"].append(test_recall)
        results["test_f1"].append(test_f1)

        if writer and example_input is not None:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)

            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc},
                               global_step=epoch)
            
            writer.add_scalars(main_tag="Recall",
                               tag_scalar_dict={"train_recall": train_recall,
                                                "test_recall": test_recall},
                               global_step=epoch)
            
            writer.add_scalars(main_tag="F1-Score",
                               tag_scalar_dict={"train_f1": train_f1,
                                                "test_f1": test_f1},
                               global_step=epoch)

            writer.add_graph(model=model,
                             input_to_model=example_input)

            writer.close()

    return results