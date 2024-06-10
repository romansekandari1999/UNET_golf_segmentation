"""
File name: golfHandler.py
Description: This script is used to handle the golf dataset.

Authors: 
Roman Sabawoon Sekandari
Frederik Hoffmann Bertelsen

"""
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class GolfDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # List all files in the images directory
        self.images = sorted([f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))])
        self.masks = sorted([f for f in os.listdir(self.masks_dir) if os.path.isfile(os.path.join(self.masks_dir, f))])

        # Debug: Print the number of images and masks
        print(f"Number of images: {len(self.images)}")
        print(f"Number of masks: {len(self.masks)}")

        self.image_mask_pairs = []
        self.images_without_masks = []

        for img_file in self.images:
            img_number = os.path.splitext(img_file)[0]
            mask_file = f"{img_number}_mask.jpg"
            if mask_file in self.masks:
                self.image_mask_pairs.append((img_file, mask_file))
            else:
                self.images_without_masks.append(img_file)
                print(f"Warning: No mask found for image {img_file}")

        # Debug: Ensure the lengths match
        if self.images_without_masks:
            raise ValueError(f"Some images do not have corresponding masks: {self.images_without_masks}")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, index):
        img_filename, mask_filename = self.image_mask_pairs[index]

        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Normalizeíng mask values to [0, 1] for binary classification
        mask = mask / 255.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask, img_filename
    
# Debugging to check if the dataset is working
if __name__ == "__main__":
    images_dir = '/content/drive/MyDrive/Colab Notebooks/project/data/golf/golfersIMG/val_images'
    masks_dir = '/content/drive/MyDrive/Colab Notebooks/project/data/golf/golfersIMG_ground_truth/val_masks'
    dataset = GolfDataset(images_dir, masks_dir)
    print(f"Dataset size: {len(dataset)}")

    image, mask, img_filename = dataset[0]
    print(f"First image filename: {img_filename}")
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

    if dataset.images_without_masks:
        print("Images without corresponding masks:")
        for img in dataset.images_without_masks:
            print(img)
    else:
        print("All images have corresponding masks.")