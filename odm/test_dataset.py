import os
import sys
import torch

# Add parent directory to path so we can import modules directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with direct paths
from odm.utils.dataset import SFPIDataset
from odm.configs.config import Config

def test_dataset():
    """Test if we can load and access the SFPI dataset"""
    config = Config()
    
    # Print paths for verification
    print(f"Images directory: {config.IMAGES_DIR}")
    print(f"Annotations directory: {config.ANNOTATIONS_DIR}")
    
    # Check if directories exist
    print(f"Images directory exists: {os.path.exists(config.IMAGES_DIR)}")
    print(f"Annotations directory exists: {os.path.exists(config.ANNOTATIONS_DIR)}")
    
    # Try to list the images directory
    if os.path.exists(config.IMAGES_DIR):
        train_dir = os.path.join(config.IMAGES_DIR, 'train')
        val_dir = os.path.join(config.IMAGES_DIR, 'val')
        test_dir = os.path.join(config.IMAGES_DIR, 'test')
        
        print(f"Train directory exists: {os.path.exists(train_dir)}")
        print(f"Val directory exists: {os.path.exists(val_dir)}")
        print(f"Test directory exists: {os.path.exists(test_dir)}")
        
        # Try to list files in train directory
        if os.path.exists(train_dir):
            try:
                files = os.listdir(train_dir)
                print(f"Number of files in train directory: {len(files)}")
                if files:
                    print(f"First few files: {files[:5]}")
            except Exception as e:
                print(f"Error listing train directory: {e}")
    
    # Try to create the dataset
    try:
        train_dataset = SFPIDataset(split='train', labeled=True, label_percentage=1)
        print(f"Train dataset length: {len(train_dataset)}")
        
        # Try to get one sample
        if len(train_dataset) > 0:
            try:
                image, target = train_dataset[0]
                print(f"Image shape: {image.shape}")
                print(f"Target keys: {target.keys()}")
                print(f"Number of boxes: {len(target['boxes'])}")
            except Exception as e:
                print(f"Error getting sample: {e}")
    except Exception as e:
        print(f"Error creating dataset: {e}")

if __name__ == "__main__":
    test_dataset() 