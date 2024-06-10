"""
File name: golf_train.py
Description: This script is used to train the UNET model on the golf dataset.

Authors: 
Roman Sabawoon Sekandari
Frederik Hoffmann Bertelsen

"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from unetSimple import UNET
from golfHandler import GolfDataset  
from golf_utils import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_metrics_to_csv,
    save_prediction_metrics_to_csv,
    plot_metrics,
    plot_prediction_metrics,
    visualize_predictions,
    Dice_Score,
    IoU,
    calculate_precision_recall,
    calculate_f1_score,
    calculate_accuracy,
    dice_loss,  
)

# Dynamic directory 
script_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
data_dir = os.path.join(project_dir, "data", "golf")
pred_dir = os.path.join(project_dir, "data", "golf_unet_results")
checkpoint_path = os.path.join(script_dir, "run3_checkpoint.pt")
CSV_FILE = os.path.join(project_dir, "data", "TF_thinPlate_SMALL_gpu_train_metrics.csv")
PRED_CSV_FILE = os.path.join(project_dir, "data", "CSV_PRED_TF_thinPlate_SMALL_gpu_prediction_metrics.csv")

# Hyperparameters used for training, can be adjusted to needs.
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 160
PIN_MEMORY = True
LOAD_MODEL = False # Set to True if you want to load a model for transfer learning

# The training function
def train_fn(loader, model, optimizer, loss_fn, scaler, additional_losses=None):
    loop = tqdm(loader)
    total_losses = {'BCE Loss': 0, 'Dice Loss': 0}
    total_metrics = {
        'Dice Score': 0,
        'IoU': 0,
        'Precision': 0,
        'Recall': 0,
        'F1 Score': 0,
        'Accuracy': 0,
        'MAE': 0,
        'MSE': 0,
    }
    total_time = 0
    total_frames = 0

    model.train()

    for batch_idx, (data, targets, _) in enumerate(loop):
        start_time = time.time()
        
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass 
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        dice_loss_value = additional_losses['Dice Loss'](predictions, targets).item()
        
        # Backward and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_losses['BCE Loss'] += loss.item()
        total_losses['Dice Loss'] += dice_loss_value

        # Calculate additional metrics
        preds = torch.sigmoid(predictions)
        preds = (preds > 0.5).float()
        mask = (targets > 0.5).float()

        intersection = torch.sum(preds * mask).item()
        total_predicted_positive = torch.sum(preds).item()
        total_actual_positive = torch.sum(mask).item()

        dice_score = Dice_Score(intersection, total_predicted_positive + total_actual_positive)
        iou = IoU(intersection, total_predicted_positive, total_actual_positive)
        precision, recall = calculate_precision_recall(preds, mask)
        f1_score = calculate_f1_score(precision, recall)
        accuracy = calculate_accuracy(preds, mask)
        mae = torch.mean(torch.abs(preds - mask)).item()
        mse = torch.mean((preds - mask) ** 2).item()

        total_metrics['Dice Score'] += dice_score
        total_metrics['IoU'] += iou
        total_metrics['Precision'] += precision
        total_metrics['Recall'] += recall
        total_metrics['F1 Score'] += f1_score
        total_metrics['Accuracy'] += accuracy
        total_metrics['MAE'] += mae
        total_metrics['MSE'] += mse

        # Calculate time taken for this batch
        end_time = time.time()
        total_time += end_time - start_time
        total_frames += data.size(0)

        # Update the tqdm loop
        loop.set_postfix(loss=loss.item())
    
    avg_losses = {key: total / len(loader) for key, total in total_losses.items()}
    avg_metrics = {key: total / len(loader) for key, total in total_metrics.items()}
    avg_fps = total_frames / total_time
    avg_metrics.update(avg_losses)
    avg_metrics['FPS'] = avg_fps
    return avg_metrics

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    additional_losses = {'Dice Loss': dice_loss}
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_loader, val_loader, test_loader = get_loaders(
        train_images_dir=os.path.join(data_dir, "golfersIMG", "train_images"),
        train_masks_dir=os.path.join(data_dir, "golfersIMG_ground_truth", "train_masks"),
        val_images_dir=os.path.join(data_dir, "golfersIMG", "val_images"),
        val_masks_dir=os.path.join(data_dir, "golfersIMG_ground_truth", "val_masks"),
        test_images_dir=os.path.join(data_dir, "golfersIMG", "test_images"),
        test_masks_dir=os.path.join(data_dir, "golfersIMG_ground_truth", "test_masks"),
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,  
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        load_checkpoint(checkpoint, model, optimizer=None)
        print("Model loaded")
        print("Let's train")
    
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True) # Patience set to 5
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        previous_lr = optimizer.param_groups[0]['lr']
        epoch_start_time = time.time()

        # Training phase metrics
        avg_train_metrics = train_fn(train_loader, model, optimizer, loss_fn, scaler, additional_losses)
        
        # Validation phase metrics
        avg_val_dice_score, avg_val_iou, avg_val_precision, avg_val_recall, avg_val_f1_score, avg_val_accuracy, avg_val_mae, avg_val_mse, avg_val_dice_loss, avg_val_bce_loss, avg_val_fps, avg_val_prediction_time, total_val_prediction_time = check_accuracy(val_loader, model, device=DEVICE)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # avg_val_dice_score for learning rate adjustment
        scheduler.step(avg_val_dice_score)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f'Learning rate decreased from {previous_lr} to {current_lr}')
        else:
            print(f'Current learning rate remains at {current_lr}')

        # Save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        # Logging the metrics to a CSV file
        metrics = [
            epoch + 1, avg_train_metrics['Dice Score'], avg_val_dice_score, avg_train_metrics['IoU'], avg_val_iou,
            avg_train_metrics['Precision'], avg_val_precision, avg_train_metrics['Recall'], avg_val_recall, avg_train_metrics['F1 Score'], avg_val_f1_score,
            avg_train_metrics['Accuracy'], avg_val_accuracy, avg_train_metrics['MAE'], avg_val_mae, avg_train_metrics['MSE'], avg_val_mse,
            epoch_time, avg_train_metrics['BCE Loss'], avg_val_bce_loss, avg_train_metrics['Dice Loss'], avg_val_dice_loss,
            avg_train_metrics['FPS'], avg_val_fps, avg_val_prediction_time, total_val_prediction_time
        ]
        save_metrics_to_csv(metrics, filename=CSV_FILE)

        # Print metrics from check accuracy. These will be logged in the CSV file aswell
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        print(f'Validation Dice Score: {avg_val_dice_score:.4f}')
        print(f'Validation IoU Score: {avg_val_iou:.4f}')
        print(f'Validation Precision: {avg_val_precision:.4f}')
        print(f'Validation Recall: {avg_val_recall:.4f}')
        print(f'Validation F1 Score: {avg_val_f1_score:.4f}')
        print(f'Validation Accuracy: {avg_val_accuracy:.4f}')
        print(f'Validation MAE: {avg_val_mae:.4f}')
        print(f'Validation MSE: {avg_val_mse:.4f}')
        print(f'Validation BCE Loss: {avg_val_bce_loss:.4f}')
        print(f'Validation Dice Loss: {avg_val_dice_loss:.4f}')
        print(f'Epoch Time: {epoch_time:.4f} seconds')
        print(f'Training Average Dice Score: {avg_train_metrics["Dice Score"]:.4f}')
        print(f'Training Average IoU: {avg_train_metrics["IoU"]:.4f}')
        print(f'Training Average Precision: {avg_train_metrics["Precision"]:.4f}')
        print(f'Training Average Recall: {avg_train_metrics["Recall"]:.4f}')
        print(f'Training Average F1 Score: {avg_train_metrics["F1 Score"]:.4f}')
        print(f'Training Average Accuracy: {avg_train_metrics["Accuracy"]:.4f}')
        print(f'Training Average MAE: {avg_train_metrics["MAE"]:.4f}')
        print(f'Training Average MSE: {avg_train_metrics["MSE"]:.4f}')
        print(f'Training Average BCE Loss: {avg_train_metrics["BCE Loss"]:.4f}')
        print(f'Training Average Dice Loss: {avg_train_metrics["Dice Loss"]:.4f}')
        print(f'Average Train FPS: {avg_train_metrics["FPS"]:.2f}')
        print(f'Average Validation FPS: {avg_val_fps:.2f}')
        print(f'Average Prediction Time: {avg_val_prediction_time:.4f} seconds')
        print(f'Total Prediction Time: {total_val_prediction_time:.4f} seconds\n')

        early_stopping(avg_val_dice_score, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Save predictions after training is complete
    print("Saving validation set predictions ...")
    save_predictions_as_imgs(
        val_loader, 
        model, 
        pred_folder=pred_dir,
        is_training=False,  # False since we are saving validation predictions
        device=DEVICE
    )
    
    # Save and plot metrics
    print("Plotting training metrics ...")
    plot_metrics(CSV_FILE)
    
    # Visualize predictions
    print("Visualizing predictions ...")
    visualize_predictions(val_loader, model, device=DEVICE, num_samples=5, is_training=False)

if __name__ == "__main__":
    main()
    print("Training done")
