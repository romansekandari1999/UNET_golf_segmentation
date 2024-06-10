"""
File name: golf_utils.py
Description: This script includes utility functions for training and evaluating the UNET model on the golf dataset.

Authors: 
Roman Sabawoon Sekandari
Frederik Hoffmann Bertelsen

"""
import os
import time
import csv
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from golfHandler import GolfDataset
from torch.utils.data import DataLoader, Subset

# Random seed for reproducibility
import numpy as np
seed = 66 # Star Wars reference :))
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dynamic directory 
script_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
pred_dir = os.path.join(project_dir, "data", "golf_unet_results")
checkpoint_path = os.path.join(project_dir, "UNET", "checkpoint.pth")
train_plot_dir = os.path.join(project_dir, "data", "plots", "train_plots")
pred_plot_dir = os.path.join(project_dir, "data", "plots", "pred_plots")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
EarlyStopping class is used to stop the training process if the model does not improve after 5 continuous non improved dice scores.

"""
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.consecutive_increases = 0

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.consecutive_increases += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.trace_func(f'Consecutive increases: {self.consecutive_increases}')
            if self.consecutive_increases >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
            self.consecutive_increases = 0

    # Saving the model if a better dice score is achieved
    def save_checkpoint(self, val_score, model):
        if self.verbose:
            self.trace_func(f'Dice score improved ({self.best_score:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save({'state_dict': model.state_dict()}, self.path)
        self.best_score = val_score

def save_checkpoint(state, filename=checkpoint_path):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

# Function to get the data loaders for training, validation, and testing
def get_loaders(
        train_images_dir,
        train_masks_dir,
        val_images_dir,
        val_masks_dir,
        test_images_dir,
        test_masks_dir,
        batch_size,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        num_workers=4,
        pin_memory=True,
):
    train_ds = GolfDataset(train_images_dir, train_masks_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    
    val_ds = GolfDataset(val_images_dir, val_masks_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    
    test_ds = GolfDataset(test_images_dir, test_masks_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader

def Dice_Score(intersection, union):
    return (2 * intersection) / (union + 1e-8)

def IoU(intersection, total_predicted_positive, total_actual_positive):
    union = total_predicted_positive + total_actual_positive - intersection
    return intersection / (union + 1e-8)

def calculate_precision_recall(preds, mask):
    TP = torch.sum(preds * mask).item()
    FP = torch.sum(preds * (1 - mask)).item()
    FN = torch.sum((1 - preds) * mask).item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return precision, recall

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def calculate_accuracy(preds, mask):
    return torch.sum(preds == mask).item() / torch.numel(preds)

def Dice_Loss(preds, targets, smooth=1e-8):
    intersection = torch.sum(preds * targets)
    return 1 - (2. * intersection + smooth) / (torch.sum(preds) + torch.sum(targets) + smooth)

def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

"""
This function is used to evaluate the model on the validation set. It calculates metrics for segmentation quality and inference times.
"""
def check_accuracy(loader, model, device):
    total_dice_score = 0
    total_iou = 0
    total_f1_score = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    total_mae = 0
    total_mse = 0
    total_dice_loss = 0
    total_time = 0
    total_frames = 0
    total_prediction_time = 0
    total_bce_loss = 0
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Evaluating", leave=True):  
            start_time = time.time() # loader start time
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).unsqueeze(1).float()

            if x.ndim == 4 and x.shape[1] != 3:
                x = x.permute(0, 3, 1, 2)

            pred_start_time = time.time()  # Start prediction timer
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            mask = (y > 0.5).float()
            pred_end_time = time.time()  # End prediction timer

            intersection = torch.sum(preds * mask).item()
            total_predicted_positive = torch.sum(preds).item()
            total_actual_positive = torch.sum(mask).item()

            dice_score = Dice_Score(intersection, total_predicted_positive + total_actual_positive)
            iou = IoU(intersection, total_predicted_positive, total_actual_positive)
            dice_loss = Dice_Loss(preds, mask).item()

            # Compute BCE Loss
            bce_loss = nn.BCEWithLogitsLoss()(model(x), y).item()

            # Calculate precision, recall, f1 score, accuracy, mae, mse
            precision, recall = calculate_precision_recall(preds, mask)
            f1_score = calculate_f1_score(precision, recall)
            accuracy = calculate_accuracy(preds, mask)
            mae = torch.mean(torch.abs(preds - mask)).item()
            mse = torch.mean((preds - mask) ** 2).item()

            total_dice_score += dice_score
            total_iou += iou
            total_f1_score += f1_score
            total_precision += precision
            total_recall += recall
            total_accuracy += accuracy
            total_mae += mae
            total_mse += mse
            total_dice_loss += dice_loss
            total_bce_loss += bce_loss

            end_time = time.time()
            total_time += end_time - start_time
            total_prediction_time += pred_end_time - pred_start_time 
            total_frames += x.size(0)

    avg_dice_score = total_dice_score / len(loader)
    avg_iou = total_iou / len(loader)
    avg_precision = total_precision / len(loader)
    avg_recall = total_recall / len(loader)
    avg_f1_score = total_f1_score / len(loader)
    avg_accuracy = total_accuracy / len(loader)
    avg_mae = total_mae / len(loader)
    avg_mse = total_mse / len(loader)
    avg_dice_loss = total_dice_loss / len(loader)
    avg_bce_loss = total_bce_loss / len(loader)
    avg_prediction_time = total_prediction_time / total_frames if total_frames > 0 else 0  
    avg_fps = total_frames/total_prediction_time

    model.train() # Set model back to training mode
    return avg_dice_score, avg_iou, avg_precision, avg_recall, avg_f1_score, avg_accuracy, avg_mae, avg_mse, avg_dice_loss, avg_bce_loss, avg_fps, avg_prediction_time, total_prediction_time

"""
This function will save the training metrics to a CSV file which can be found in the data folder.
"""
def save_metrics_to_csv(metrics, filename="metrics.csv", mode='a'):
    file_exists = os.path.isfile(filename)
    with open(filename, mode=mode) as f:
        writer = csv.writer(f)
        if not file_exists or mode == 'w':
            writer.writerow([
                "Epoch", "Train Dice Score", "Val Dice Score", "Train IoU", "Val IoU", "Train Precision", "Val Precision",
                "Train Recall", "Val Recall", "Train F1 Score", "Val F1 Score", "Train Accuracy", "Val Accuracy",
                "Train MAE", "Val MAE", "Train MSE", "Val MSE", "Epoch Time", "Train BCE Loss", "Val BCE Loss",
                "Train Dice Loss", "Val Dice Loss", "Avg Train FPS", "Avg Val FPS", "Avg Prediction Time", "Total Prediction Time"
            ])
        writer.writerow(metrics)

"""
This function will save the prediction metrics to a CSV file which can be found in the data folder.
"""
def save_prediction_metrics_to_csv(metrics, filename="prediction_metrics.csv", mode='w'):
    with open(filename, mode=mode) as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dice Score", "IoU", "Precision", "Recall", "F1 Score", "Accuracy",
            "MAE", "MSE", "BCE Loss", "Dice Loss", "Prediction Time"
        ])
        for metric in metrics:
            writer.writerow(metric)

def plot_single_metric(values, label, xlabel, ylabel, title, save_path):
    plt.figure()
    plt.plot(values, label=label, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

"""
This function is used to plot the metrics from the CSV file, in order to compare training and validation metrics.
"""
def plot_comparison_metric(train_values, val_values, label, xlabel, ylabel, title, save_path):
    plt.figure()
    plt.plot(train_values, label=f"Train {label}", color='blue')
    plt.plot(val_values, label=f"Val {label}", color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

"""
This function is used to plot the metrics from the CSV file.
"""
def plot_metrics(csv_filename):
    plot_dir = train_plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    metrics = {
        "epochs": [],
        "train_dice_scores": [],
        "val_dice_scores": [],
        "train_ious": [],
        "val_ious": [],
        "train_precisions": [],
        "val_precisions": [],
        "train_recalls": [],
        "val_recalls": [],
        "train_f1_scores": [],
        "val_f1_scores": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "train_maes": [],
        "val_maes": [],
        "train_mses": [],
        "val_mses": [],
        "epoch_times": [],
        "train_bce_losses": [],
        "val_bce_losses": [],
        "train_dice_losses": [],
        "val_dice_losses": [],
        "avg_train_fps": [],
        "avg_val_fps": [],
        "avg_prediction_times": [],
        "total_prediction_times": []
    }

    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics["epochs"].append(int(row["Epoch"]))
            metrics["train_dice_scores"].append(float(row["Train Dice Score"]))
            metrics["val_dice_scores"].append(float(row["Val Dice Score"]))
            metrics["train_ious"].append(float(row["Train IoU"]))
            metrics["val_ious"].append(float(row["Val IoU"]))
            metrics["train_precisions"].append(float(row["Train Precision"]))
            metrics["val_precisions"].append(float(row["Val Precision"]))
            metrics["train_recalls"].append(float(row["Train Recall"]))
            metrics["val_recalls"].append(float(row["Val Recall"]))
            metrics["train_f1_scores"].append(float(row["Train F1 Score"]))
            metrics["val_f1_scores"].append(float(row["Val F1 Score"]))
            metrics["train_accuracies"].append(float(row["Train Accuracy"]))
            metrics["val_accuracies"].append(float(row["Val Accuracy"]))
            metrics["train_maes"].append(float(row["Train MAE"]))
            metrics["val_maes"].append(float(row["Val MAE"]))
            metrics["train_mses"].append(float(row["Train MSE"]))
            metrics["val_mses"].append(float(row["Val MSE"]))
            metrics["epoch_times"].append(float(row["Epoch Time"]))
            metrics["train_bce_losses"].append(float(row["Train BCE Loss"]))
            metrics["val_bce_losses"].append(float(row["Val BCE Loss"]))
            metrics["train_dice_losses"].append(float(row["Train Dice Loss"]))
            metrics["val_dice_losses"].append(float(row["Val Dice Loss"]))
            metrics["avg_train_fps"].append(float(row["Avg Train FPS"]))
            metrics["avg_val_fps"].append(float(row["Avg Val FPS"]))
            metrics["avg_prediction_times"].append(float(row["Avg Prediction Time"]))
            metrics["total_prediction_times"].append(float(row["Total Prediction Time"]))

    plot_comparison_metric(metrics["train_dice_scores"], metrics["val_dice_scores"], "Dice Score", "Epoch", "Dice Score", "Dice Score per Epoch", os.path.join(plot_dir, "dice_score_per_epoch.png"))
    plot_comparison_metric(metrics["train_ious"], metrics["val_ious"], "IoU", "Epoch", "IoU", "IoU per Epoch", os.path.join(plot_dir, "iou_per_epoch.png"))
    plot_comparison_metric(metrics["train_precisions"], metrics["val_precisions"], "Precision", "Epoch", "Precision", "Precision per Epoch", os.path.join(plot_dir, "precision_per_epoch.png"))
    plot_comparison_metric(metrics["train_recalls"], metrics["val_recalls"], "Recall", "Epoch", "Recall", "Recall per Epoch", os.path.join(plot_dir, "recall_per_epoch.png"))
    plot_comparison_metric(metrics["train_f1_scores"], metrics["val_f1_scores"], "F1 Score", "Epoch", "F1 Score", "F1 Score per Epoch", os.path.join(plot_dir, "f1_score_per_epoch.png"))
    plot_comparison_metric(metrics["train_accuracies"], metrics["val_accuracies"], "Accuracy", "Epoch", "Accuracy", "Accuracy per Epoch", os.path.join(plot_dir, "accuracy_per_epoch.png"))
    plot_comparison_metric(metrics["train_maes"], metrics["val_maes"], "MAE", "Epoch", "MAE", "MAE per Epoch", os.path.join(plot_dir, "mae_per_epoch.png"))
    plot_comparison_metric(metrics["train_mses"], metrics["val_mses"], "MSE", "Epoch", "MSE", "MSE per Epoch", os.path.join(plot_dir, "mse_per_epoch.png"))
    plot_single_metric(metrics["epoch_times"], "Epoch Time", "Epoch", "Time (s)", "Epoch Time per Epoch", os.path.join(plot_dir, "epoch_time_per_epoch.png"))
    plot_comparison_metric(metrics["train_bce_losses"], metrics["val_bce_losses"], "BCE Loss", "Epoch", "BCE Loss", "BCE Loss per Epoch", os.path.join(plot_dir, "bce_loss_per_epoch.png"))
    plot_comparison_metric(metrics["train_dice_losses"], metrics["val_dice_losses"], "Dice Loss", "Epoch", "Dice Loss", "Dice Loss per Epoch", os.path.join(plot_dir, "dice_loss_per_epoch.png"))
    plot_single_metric(metrics["avg_train_fps"], "Avg Train FPS", "Epoch", "FPS", "Average Training FPS per Epoch", os.path.join(plot_dir, "avg_train_fps_per_epoch.png"))
    plot_single_metric(metrics["avg_val_fps"], "Avg Val FPS", "Epoch", "FPS", "Average Validation FPS per Epoch", os.path.join(plot_dir, "avg_val_fps_per_epoch.png"))
    plot_single_metric(metrics["avg_prediction_times"], "Avg Prediction Time", "Epoch", "Avg Prediction Time (s)", "Average Prediction Time per Epoch", os.path.join(plot_dir, "avg_prediction_time_per_epoch.png"))
    plot_single_metric(metrics["total_prediction_times"], "Total Prediction Time", "Epoch", "Total Prediction Time (s)", "Total Prediction Time per Epoch", os.path.join(plot_dir, "total_prediction_time_per_epoch.png"))

  
def plot_prediction_metrics(csv_filename):
    plot_dir = pred_plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    metrics = {
        "dice_scores": [],
        "ious": [],
        "precisions": [],
        "recalls": [],
        "f1_scores": [],
        "accuracies": [],
        "maes": [],
        "mses": [],
        "bce_losses": [],
        "dice_losses": [],
        "prediction_times": []
    }

    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics["dice_scores"].append(float(row["Dice Score"]))
            metrics["ious"].append(float(row["IoU"]))
            metrics["precisions"].append(float(row["Precision"]))
            metrics["recalls"].append(float(row["Recall"]))
            metrics["f1_scores"].append(float(row["F1 Score"]))
            metrics["accuracies"].append(float(row["Accuracy"]))
            metrics["maes"].append(float(row["MAE"]))
            metrics["mses"].append(float(row["MSE"]))
            metrics["bce_losses"].append(float(row["BCE Loss"]))
            metrics["dice_losses"].append(float(row["Dice Loss"]))
            metrics["prediction_times"].append(float(row["Prediction Time"]))

    num_images = len(metrics["dice_scores"])
    image_indices = list(range(1, num_images + 1))

    plot_single_metric(metrics["dice_scores"], "Dice Score", "Test Images", "Dice Score", "Dice Score per Test Image", os.path.join(plot_dir, "dice_score_per_image.png"))
    plot_single_metric(metrics["ious"], "IoU", "Test Images", "IoU", "IoU per Test Image", os.path.join(plot_dir, "iou_per_image.png"))
    plot_single_metric(metrics["precisions"], "Precision", "Test Images", "Precision", "Precision per Test Image", os.path.join(plot_dir, "precision_per_image.png"))
    plot_single_metric(metrics["recalls"], "Recall", "Test Images", "Recall", "Recall per Test Image", os.path.join(plot_dir, "recall_per_image.png"))
    plot_single_metric(metrics["f1_scores"], "F1 Score", "Test Images", "F1 Score", "F1 Score per Test Image", os.path.join(plot_dir, "f1_score_per_image.png"))
    plot_single_metric(metrics["accuracies"], "Accuracy", "Test Images", "Accuracy", "Accuracy per Test Image", os.path.join(plot_dir, "accuracy_per_image.png"))
    plot_single_metric(metrics["maes"], "MAE", "Test Images", "MAE", "MAE per Test Image", os.path.join(plot_dir, "mae_per_image.png"))
    plot_single_metric(metrics["mses"], "MSE", "Test Images", "MSE", "MSE per Test Image", os.path.join(plot_dir, "mse_per_image.png"))
    plot_single_metric(metrics["bce_losses"], "BCE Loss", "Test Images", "BCE Loss", "BCE Loss per Test Image", os.path.join(plot_dir, "bce_loss_per_image.png"))
    plot_single_metric(metrics["dice_losses"], "Dice Loss", "Test Images", "Dice Loss", "Dice Loss per Test Image", os.path.join(plot_dir, "dice_loss_per_image.png"))
    plot_single_metric(metrics["prediction_times"], "Prediction Time", "Test Images", "Prediction Time (s)", "Prediction Time per Test Image", os.path.join(plot_dir, "prediction_time_per_image.png"))

"""
This function saves the predictions as images in the pred_dir folder.
"""
def save_predictions_as_imgs(loader, model, pred_folder=pred_dir, is_training=True, device="cuda"):
    model.eval()
    saved_folder = os.path.join(pred_folder, "train_predictions" if is_training else "test_predictions")
    os.makedirs(saved_folder, exist_ok=True)
    for idx, (x, y, original_filename) in enumerate(loader):
        print(f"Saving predictions for: {original_filename}")
        if x.ndim == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2) 

        x = x.to(device=device).float()
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        filename = os.path.splitext(os.path.basename(original_filename[0]))[0]
        save_path = os.path.join(saved_folder, f"pred_{filename}.png")
        torchvision.utils.save_image(preds, save_path)
    
    model.train()

"""
This function will create sample predictions for visualization purposes.
"""
def visualize_predictions(loader, model, device="cuda", num_samples=5, is_training=False):
    plot_dir = train_plot_dir if is_training else pred_plot_dir
    random_state = 66 
    model.eval()
    random.seed(random_state)
    samples = random.sample(list(loader), num_samples)
    saved_paths = []

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for idx, (x, y, original_filename) in enumerate(samples):
        print(f"Sample {idx + 1}: {original_filename}")
        
        if x.ndim == 4:
            # If the input tensor is 4D (batch of images), we need to select one image
            x = x[0]
            y = y[0]
        
        if x.ndim == 3 and x.shape[0] != 3:
            x = x.permute(1, 2, 0)  # We convert from [C, H, W] to [H, W, C]
        
        x = x.to(device=device).float()
        y = y.to(device=device).float()

        with torch.no_grad():
            preds = torch.sigmoid(model(x.unsqueeze(0))) 
            preds = (preds > 0.5).float()

        x = x.cpu().numpy().transpose(1, 2, 0)  # Ensure shape is [H, W, C]
        x = (x - x.min()) / (x.max() - x.min())
        preds = preds.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(x)
        ax[0].set_title('Input Image')
        ax[1].imshow(y, cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(preds, cmap='gray')
        ax[2].set_title('Predicted Mask')

        fig.suptitle(f'Sample {idx + 1}: {original_filename}')
        plt.tight_layout()
        pred_path = os.path.join(plot_dir, f"prediction_sample_{idx + 1}.png")
        plt.savefig(pred_path)
        plt.close()
        saved_paths.append(pred_path)

    model.train()
    return saved_paths
