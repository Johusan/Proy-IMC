import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# Constants for image preprocessing
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMG_STD = [0.229, 0.224, 0.225]   # ImageNet std


class CustomResize:
    """Custom resize transform that maintains aspect ratio"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        elif isinstance(img, Image.Image):
            w, h = img.size
        else:  # numpy array
            h, w = img.shape[:2]
            
        scale = max(w, h) / float(self.size)
        new_w, new_h = int(w / scale), int(h / scale)
        return transforms.functional.resize(img, (new_h, new_w))


class BMIDataset(Dataset):
    """Dataset class for BMI prediction from images"""
    def __init__(self, dataframe, image_col='image_path'):
        self.df = dataframe
        self.image_col = image_col
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            CustomResize(IMG_SIZE),
            transforms.Pad(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        image_path = row[self.image_col]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        
        # Get BMI value (if available)
        bmi = row.get('bmi', None)
        
        return image, bmi, row.get('individual_id', idx)


def load_sample_data(data_dir="data"):
    """
    Load sample data for BMI prediction.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        DataFrame with image paths and metadata
    """
    csv_path = os.path.join(data_dir, "visual-body-to-bmi.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset CSV not found at {csv_path}")
        print("Please download the sample data from the authors:")
        print("Contact: rmanichand@ethz.ch or planger@ethz.ch")
        return None
    
    # Load the full dataset
    df = pd.read_csv(csv_path)
    
    # Get list of available folders
    available_folders = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
    
    # Filter dataframe to only include images from available folders
    df['folder'] = df['individual_id'].astype(str)
    sample_data = df[df['folder'].isin(available_folders)].copy()
    
    # Update image paths to match our local structure
    sample_data['image_path'] = sample_data.apply(
        lambda row: os.path.join(data_dir, row['folder'], os.path.basename(row['image_path'])),
        axis=1
    )
    
    # Verify images exist and keep only those that do
    sample_data = sample_data[sample_data['image_path'].apply(os.path.exists)].reset_index(drop=True)
    
    if len(sample_data) == 0:
        print("No valid images found. Please check your data directory structure.")
        print("Expected structure:")
        print("  data/")
        print("    visual-body-to-bmi.csv")
        print("    individual_id_folders/")
        print("      *.jpg or *.png files")
        return None
    
    print(f"Found {len(sample_data)} valid images for testing")
    return sample_data 