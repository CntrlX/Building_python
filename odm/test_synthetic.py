import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odm.configs.config import Config
from odm.utils.dataset import SFPIDataset, get_data_loaders
from odm.model.mask_rcnn import TeacherStudentMaskRCNN

def visualize_sample(image, target, idx=0):
    """
    Visualize an image with its boxes and masks
    
    Args:
        image (Tensor): The image tensor
        target (dict): The target dictionary with boxes, labels, and masks
        idx (int): Sample index for display purposes
    """
    # Convert image tensor to numpy array
    img_np = image.permute(1, 2, 0).numpy()
    # Denormalize
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot image with boxes
    ax[0].imshow(img_np)
    ax[0].set_title(f"Sample {idx} - Image with Boxes")
    
    # Get boxes and labels
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    
    # Draw boxes and labels
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        ax[0].add_patch(rect)
        ax[0].text(x1, y1, f"Class {label}", bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Plot masks if available
    if 'masks' in target:
        masks = target['masks'].numpy()
        # Combine all masks into a single image
        mask_np = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.float32)
        for i, mask in enumerate(masks):
            # Add mask with different colors
            mask_np += mask * (i + 1)
        
        ax[1].imshow(mask_np, cmap='jet')
        ax[1].set_title(f"Sample {idx} - Masks")
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(Config().ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"))
    plt.close()

def test_dataset():
    """Test the dataset by loading a few samples and visualizing them"""
    print("Testing dataset with synthetic annotations...")
    
    # Create dataset
    dataset = SFPIDataset(split='train', use_synthetic_annotations=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a few samples
    for i in range(min(3, len(dataset))):
        try:
            image, target = dataset[i]
            print(f"Sample {i}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Boxes shape: {target['boxes'].shape}")
            print(f"  Labels: {target['labels']}")
            
            # Visualize sample
            visualize_sample(image, target, i)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    
    print("Dataset test completed. Check output directory for visualizations.")

def test_data_loaders():
    """Test data loaders to ensure they return valid batches"""
    print("Testing data loaders...")
    
    # Create data loaders
    config = Config()
    data_loaders = get_data_loaders(config, use_synthetic_annotations=True)
    
    # Test labeled data loader
    print("Testing labeled data loader...")
    labeled_loader = data_loaders['train_labeled']
    print(f"Labeled loader size: {len(labeled_loader.dataset)}")
    
    # Load a batch
    for i, (images, targets) in enumerate(labeled_loader):
        if i >= 1:
            break
        
        print(f"Batch {i}:")
        print(f"  Images count: {len(images)}")
        print(f"  Targets count: {len(targets)}")
        
        # Check if any targets have boxes
        has_boxes = False
        for j, target in enumerate(targets):
            print(f"  Target {j} boxes shape: {target['boxes'].shape}")
            if len(target['boxes']) > 0:
                has_boxes = True
        
        print(f"  Batch has boxes: {has_boxes}")
        
        # Visualize first image in batch
        if len(images) > 0:
            visualize_sample(images[0], targets[0], f"batch_{i}_sample_0")
    
    print("Data loader test completed. Check output directory for visualizations.")

def test_model():
    """Test the model by running a forward pass on a sample batch"""
    print("Testing model...")
    
    # Create data loaders
    config = Config()
    data_loaders = get_data_loaders(config, use_synthetic_annotations=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = TeacherStudentMaskRCNN(config.NUM_CLASSES)
    model.to(device)
    
    # Set model to training mode
    model.student.train()
    
    # Load a batch
    try:
        images, targets = next(iter(data_loaders['train_labeled']))
        
        # Check if any targets have boxes
        has_boxes = False
        for target in targets:
            if len(target['boxes']) > 0:
                has_boxes = True
                break
        
        print(f"Batch has boxes: {has_boxes}")
        
        if not has_boxes:
            print("ERROR: No boxes found in batch. Model test will likely fail.")
            # Add synthetic box to first target
            if len(images) > 0 and len(targets) > 0:
                print("Adding synthetic box to first target...")
                image = images[0]
                _, height, width = image.shape
                boxes = torch.tensor([[10.0, 10.0, width/2, height/2]], dtype=torch.float32)
                labels = torch.tensor([1], dtype=torch.int64)
                masks = torch.zeros((1, height, width), dtype=torch.uint8)
                masks[0, 10:int(height/2), 10:int(width/2)] = 1
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                
                targets[0]['boxes'] = boxes
                targets[0]['labels'] = labels
                targets[0]['masks'] = masks
                targets[0]['area'] = area
                targets[0]['iscrowd'] = torch.zeros((1,), dtype=torch.int64)
        
        # Move data to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        print(f"Model forward pass successful!")
        print(f"Loss dict: {loss_dict}")
        print(f"Total loss: {losses.item()}")
    except Exception as e:
        print(f"Error testing model: {e}")
    
    print("Model test completed.")

def main():
    """Run all tests"""
    print("Running synthetic annotation tests...")
    
    # Test dataset
    test_dataset()
    
    # Test data loaders
    test_data_loaders()
    
    # Test model
    test_model()
    
    print("All tests completed!")

if __name__ == "__main__":
    main() 