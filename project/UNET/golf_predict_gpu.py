"""
File name: golf_predict_gpu.py
Description: This script is used to predict the UNET model on the golf dataset.

Authors: 
Roman Sabawoon Sekandari
Frederik Hoffmann Bertelsen

"""
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from unetSimple import UNET
from golf_utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    save_prediction_metrics_to_csv,
    plot_prediction_metrics,
    visualize_predictions,
)

# Dynamic directory 
script_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, os.pardir)) 
data_dir = os.path.join(project_dir, "data", "golf")
pred_dir = os.path.join(project_dir, "data", "golf_unet_results")
checkpoint_path = os.path.join(script_dir, "TF_run_6checkpoint.pt") # Path should be changed accoring to the model checkpoint
CSV_FILE = os.path.join(project_dir, "data", "prediction_metrics.csv") # Will save the prediction metrics in the data folder

# Hyperparameters
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 
IMAGE_WIDTH = 160 
PIN_MEMORY = True
LOAD_MODEL = True

def main():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # We only need the test loader
    _, _, test_loader = get_loaders(
        train_images_dir=os.path.join(data_dir, "golfersIMG", "train_images"),
        train_masks_dir=os.path.join(data_dir, "golfersIMG_ground_truth", "train_masks"),
        val_images_dir=os.path.join(data_dir, "golfersIMG", "val_images"),
        val_masks_dir=os.path.join(data_dir, "golfersIMG_ground_truth", "val_masks"),
        test_images_dir=os.path.join(data_dir, "golfersIMG", "test_images"),
        test_masks_dir=os.path.join(data_dir, "golfersIMG_ground_truth", "test_masks"),
        batch_size=BATCH_SIZE,
        test_transform=test_transform, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Load the pre-trained model
    if LOAD_MODEL:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        load_checkpoint(checkpoint, model, optimizer=None)
    print("Model loaded")
    print("Let's predict")
    
    # Evaluate metrics on the test set
    print("Starting prediction...")
    avg_dice_score, avg_iou, avg_precision, avg_recall, avg_f1_score, avg_accuracy, avg_mae, avg_mse, avg_dice_loss, avg_bce_loss, avg_fps, avg_prediction_time, total_prediction_time = check_accuracy(test_loader, model, device=DEVICE)
    print(f'Test Dice Score: {avg_dice_score:.4f}')
    print(f'Test IoU Score: {avg_iou:.4f}')
    print(f'Test Precision: {avg_precision:.4f}')
    print(f'Test Recall: {avg_recall:.4f}')
    print(f'Test F1 Score: {avg_f1_score:.4f}')
    print(f'Test Accuracy: {avg_accuracy:.4f}')
    print(f'Test MAE: {avg_mae:.4f}')
    print(f'Test MSE: {avg_mse:.4f}')
    print(f'Test Dice Loss: {avg_dice_loss:.4f}')
    print(f'Test BCE Loss: {avg_bce_loss:.4f}')
    print(f'Test FPS: {avg_fps:.2f}')
    print(f'Test Average Prediction Time: {avg_prediction_time:.4f} seconds')
    print(f'Test Total Prediction Time: {total_prediction_time:.4f} seconds')
    
    print("Saving metrics to CSV")
    # Save the metrics to CSV
    save_prediction_metrics_to_csv(
        [
            [avg_dice_score, avg_iou, avg_precision, avg_recall, avg_f1_score, avg_accuracy, avg_mae, avg_mse, avg_bce_loss, avg_dice_loss, avg_prediction_time, avg_fps]
        ],
        filename=CSV_FILE
    )
    
    print("Plotting metrics from CSV")
    # Plot the metrics
    plot_prediction_metrics(CSV_FILE)

    # Visualize random predictions
    print("Getting samples")
    pred_paths = visualize_predictions(test_loader, model, device=DEVICE, num_samples=5, is_training=False)

    print("Saving test set predictions ...")
    save_predictions_as_imgs(
       test_loader,
       model,
       pred_folder=pred_dir,
       is_training=False,
       device=DEVICE
    )
        
if __name__ == "__main__":
    main()
    print("Prediction completed successfully!")
