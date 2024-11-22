import torch
from sklearn.metrics import precision_score as sk_precision_score

def calculate_metrics(outputs, targets):
    _, predicted = torch.max(outputs, 1)  # Get the predicted class
    accuracy = (predicted == targets).float().mean().item()  # Calculate accuracy
    precision = sk_precision_score(targets, predicted, average='weighted', zero_division=0)  # Calculate precision
    return accuracy, precision