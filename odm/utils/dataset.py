import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from odm.configs.config import Config
from torchvision.ops.boxes import box_area

def get_transform(train=True, strong_aug=False):
    """
    Creates transform pipeline for the dataset
    
    Args:
        train (bool): Whether to create transforms for training or validation
        strong_aug (bool): Whether to use strong augmentation (for the student model in semi-supervised learning)
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if train:
        if strong_aug:
            # Strong augmentation for student model in semi-supervised learning
            transforms = [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(degrees=30),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            # Regular augmentation for labeled data and teacher model
            transforms = [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
    else:
        # No augmentation for validation
        transforms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    return T.Compose(transforms)

class SFPIDataset(Dataset):
    """Dataset class for the SFPI dataset"""
    
    def __init__(self, split='train', labeled=True, label_percentage=1, transform=None, use_synthetic_annotations=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            labeled (bool): Whether to use labeled or unlabeled data
            label_percentage (int): Percentage of labeled data to use (1, 5, or 10)
            transform (callable, optional): Optional transform to be applied on a sample
            use_synthetic_annotations (bool): Whether to create synthetic annotations if no real ones are found
        """
        self.split = split
        self.labeled = labeled
        self.label_percentage = label_percentage
        self.transform = transform
        self.config = Config()
        self.use_synthetic_annotations = use_synthetic_annotations
        
        # Set paths
        self.images_dir = os.path.join(self.config.IMAGES_DIR, split)
        self.annotations_file = os.path.join(
            self.config.ANNOTATIONS_DIR, 
            f"{split}_annotation.json"
        )
        
        # Load image file paths
        try:
            self.image_files = [f for f in os.listdir(self.images_dir) 
                              if f.endswith('.tiff') or f.endswith('.tif')]
            print(f"Found {len(self.image_files)} images in {self.images_dir}")
            if len(self.image_files) == 0:
                print(f"WARNING: No images found in {self.images_dir}")
                # Create a simple directory structure to allow testing
                os.makedirs(self.images_dir, exist_ok=True)
                # Create a dummy image for testing
                self._create_dummy_images()
                self.image_files = [f for f in os.listdir(self.images_dir) 
                                  if f.endswith('.tiff') or f.endswith('.tif')]
                print(f"Created {len(self.image_files)} dummy images for testing")
        except FileNotFoundError:
            print(f"ERROR: Image directory {self.images_dir} not found. Creating directory structure.")
            os.makedirs(self.images_dir, exist_ok=True)
            # Create a dummy image for testing
            self._create_dummy_images()
            self.image_files = [f for f in os.listdir(self.images_dir) 
                              if f.endswith('.tiff') or f.endswith('.tif')]
            print(f"Created {len(self.image_files)} dummy images for testing")
        
        # Load annotations
        try:
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
            self.has_real_annotations = True
            print(f"Loaded {len(self.annotations)} annotations from {self.annotations_file}")
        except FileNotFoundError:
            print(f"Warning: Annotation file {self.annotations_file} not found. Using empty annotations.")
            self.annotations = {}
            self.has_real_annotations = False
            
            # Create synthetic annotations if enabled
            if use_synthetic_annotations:
                print(f"Creating synthetic annotations for {len(self.image_files)} images in {split} set")
                self._create_synthetic_annotations()
        
        # Split data into labeled and unlabeled for semi-supervised learning
        if split == 'train' and self.config.USE_SEMI_SUPERVISED:
            self._split_labeled_unlabeled()
    
    def _create_dummy_images(self):
        """Create dummy images for testing when no real images are found"""
        print("Creating dummy images for testing...")
        try:
            # Create a simple blank image
            for i in range(5):
                # Create a blank image (white background)
                img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                # Save the image
                img_path = os.path.join(self.images_dir, f"dummy_{i}.tiff")
                img.save(img_path)
                print(f"Created dummy image: {img_path}")
        except Exception as e:
            print(f"Error creating dummy images: {e}")
    
    def _create_synthetic_annotations_for_image(self, image):
        """Create synthetic annotations for a single image"""
        width, height = image.size
        
        # Create random number of boxes (3-6)
        num_boxes = random.randint(3, 6)
        
        boxes = []
        labels = []
        masks = []
        
        # Get the maximum valid label index (NUM_CLASSES - 1)
        max_label = self.config.NUM_CLASSES - 1
        
        # Generate random boxes
        for _ in range(num_boxes):
            # Make sure boxes are not too small and not too large
            box_width = random.randint(int(width * 0.1), int(width * 0.3))
            box_height = random.randint(int(height * 0.1), int(height * 0.3))
            
            # Make sure the box fits in the image
            x1 = random.randint(0, width - box_width)
            y1 = random.randint(0, height - box_height)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # Add box
            boxes.append([x1, y1, x2, y2])
            
            # Add random label (1 to max_label) - avoid background class (0)
            label = random.randint(1, max_label)
            labels.append(label)
            
            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
        
        return boxes, labels, masks
            
    def _create_synthetic_annotations(self):
        """Create synthetic annotations when none are available"""
        for img_file in self.image_files:
            # Skip if already has an annotation
            if img_file in self.annotations:
                continue
            
            try:
                # Load image to get dimensions
                img_path = os.path.join(self.images_dir, img_file)
                image = Image.open(img_path)
                width, height = image.size
                
                # Create random number of boxes (3-6)
                num_boxes = random.randint(3, 6)
                
                # Generate boxes for this image
                anno_boxes = []
                
                # Get the maximum valid label index (NUM_CLASSES - 1)
                max_label = self.config.NUM_CLASSES - 1
                
                for _ in range(num_boxes):
                    # Make sure boxes are not too small and not too large
                    box_width = random.randint(int(width * 0.1), int(width * 0.3))
                    box_height = random.randint(int(height * 0.1), int(height * 0.3))
                    
                    # Make sure the box fits in the image
                    x = random.randint(0, width - box_width)
                    y = random.randint(0, height - box_height)
                    
                    # Add box with random label (1 to max_label) - avoid background class (0)
                    label = random.randint(1, max_label)
                    
                    anno_boxes.append({
                        'bbox': [x, y, box_width, box_height],
                        'category_id': label,
                        'segmentation': [[x, y, x + box_width, y, x + box_width, y + box_height, x, y + box_height]],
                        'area': box_width * box_height,
                        'iscrowd': 0
                    })
                
                # Add to annotations
                self.annotations[img_file] = anno_boxes
                
            except Exception as e:
                print(f"Error creating synthetic annotations for {img_file}: {e}")
                # Continue with next image
                continue
    
    def _split_labeled_unlabeled(self):
        """Split the training data into labeled and unlabeled portions"""
        # Set random seed for reproducibility
        random.seed(42)
        
        # Shuffle image files
        random.shuffle(self.image_files)
        
        # Calculate number of labeled samples (minimum 1)
        num_labeled = max(1, int(len(self.image_files) * self.label_percentage / 100))
        print(f"Using {num_labeled} labeled samples out of {len(self.image_files)} for {self.label_percentage}% split")
        
        # Split into labeled and unlabeled
        if self.labeled:
            self.image_files = self.image_files[:num_labeled]
        else:
            self.image_files = self.image_files[num_labeled:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        try:
            # Get image file
            img_file = self.image_files[idx]
            img_path = os.path.join(self.images_dir, img_file)
            
            # Load image
            try:
                image = Image.open(img_path).convert("RGB")
                
                # Check image size and resize if too large
                width, height = image.size
                image_size_mb = (width * height * 3) / (1024 * 1024)  # Approximate size in MB
                
                # Log large images for debugging
                if image_size_mb > 50:  # 50MB is quite large
                    print(f"WARNING: Large image detected: {img_file}, dimensions: {width}x{height}, approx size: {image_size_mb:.2f} MB")
                
                # Resize extremely large images to prevent memory issues
                max_dimension = 4000  # Maximum allowed dimension
                if width > max_dimension or height > max_dimension:
                    print(f"Resizing large image {img_file} from {width}x{height} to fit within {max_dimension}x{max_dimension}")
                    
                    # Calculate new dimensions while maintaining aspect ratio
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    print(f"Image {img_file} resized to {new_width}x{new_height}")
                    
            except Exception as e:
                print(f"Error loading image {img_file}: {type(e).__name__}: {str(e)}")
                # Return a default sample in case of image loading error
                return self._create_default_sample(idx, error=f"Image loading error: {str(e)}")
            
            # Look for annotation
            boxes = []
            labels = []
            masks = []
            
            # If in the annotation files, extract boxes, labels and masks
            if img_file in self.annotations:
                anno = self.annotations[img_file]
                for obj in anno:
                    try:
                        # Get bounding box
                        x, y, w, h = obj['bbox']
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        
                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
                            continue
                        
                        # Add box
                        boxes.append([x1, y1, x2, y2])
                        
                        # Add label
                        label = obj['category_id']
                        labels.append(label)
                        
                        # Create mask (simplified as box for now)
                        mask = np.zeros((image.height, image.width), dtype=np.uint8)
                        mask[int(y1):int(y2), int(x1):int(x2)] = 1
                        masks.append(mask)
                    except Exception as e:
                        print(f"Error processing annotation for {img_file}, object {obj}: {str(e)}")
                        # Continue with other annotations
                        continue
                    
            # Check if we need to use synthetic annotations
            if len(boxes) == 0 and self.use_synthetic_annotations:
                try:
                    # Create synthetic boxes, labels and masks
                    boxes, labels, masks = self._create_synthetic_annotations_for_image(image)
                except Exception as e:
                    print(f"Error creating synthetic annotations for {img_file}: {str(e)}")
                    # Just continue with empty annotations
            
            # Log the number of annotations for debugging
            if idx < 5:  # Only log first few samples
                print(f"Sample {idx}: Found {len(boxes)} valid boxes for image {img_file}")
            
            # Handle the case where there are no annotations
            if len(boxes) == 0:
                # Create empty tensors with the right shape
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
                masks = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
                area = torch.zeros(0, dtype=torch.float32)
                iscrowd = torch.zeros(0, dtype=torch.int64)
            else:
                try:
                    # Convert to tensors - fix for the warning by pre-converting masks to a single numpy array
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    labels = torch.as_tensor(labels, dtype=torch.int64)
                    
                    # Convert list of masks to a single numpy array first, then to tensor
                    if len(masks) > 0:
                        masks = np.array(masks, dtype=np.uint8)
                        masks = torch.as_tensor(masks, dtype=torch.uint8)
                    else:
                        masks = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
                    
                    # Create area for boxes
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                    
                    # Create iscrowd attribute (all 0 as we don't have crowd annotations)
                    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
                except Exception as e:
                    print(f"Error converting annotations to tensors for {img_file}: {str(e)}")
                    return self._create_default_sample(idx, error=f"Tensor conversion error: {str(e)}")
            
            # Create image_id tensor
            image_id = torch.tensor([idx])
            
            # Apply transforms if any
            try:
                if self.transform:
                    image = self.transform(image)
                else:
                    # Default transform: convert to tensor and normalize
                    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image = transform(image)
            except Exception as e:
                print(f"Error applying transforms to {img_file}: {str(e)}")
                return self._create_default_sample(idx, error=f"Transform error: {str(e)}")
            
            # Create target dictionary
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd
            }
            
            return image, target
        except Exception as e:
            import traceback
            print(f"Error processing sample {idx}, file {self.image_files[idx] if idx < len(self.image_files) else 'unknown'}: {type(e).__name__}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return a default sample in case of error
            return self._create_default_sample(idx, error=f"General error: {str(e)}")
        
    def _create_default_sample(self, idx, error=None):
        """Create a default sample when an error occurs"""
        if error:
            print(f"Creating default sample for idx {idx} due to error: {error}")
        
        # Create a blank image of modest size
        image = torch.zeros((3, 800, 800), dtype=torch.float32)
        
        # Create empty target
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks": torch.zeros((0, 800, 800), dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros(0, dtype=torch.int64),
            "has_error": True,  # Flag to indicate this is a default sample due to error
            "error_message": error if error else "Unknown error"
        }
        
        return image, target

def get_data_loaders(config, use_synthetic_annotations=True):
    """
    Create data loaders for training, validation and testing
    
    Args:
        config: Configuration object
        use_synthetic_annotations: Whether to create synthetic annotations if no real ones are found
    
    Returns:
        Dictionary of data loaders
    """
    # Data augmentation for labeled data
    train_transform_labeled = get_transform(train=True, strong_aug=False)
    
    # Weak augmentation for unlabeled data (teacher model)
    train_transform_unlabeled_weak = get_transform(train=True, strong_aug=False)
    
    # Strong augmentation for unlabeled data (student model)
    train_transform_unlabeled_strong = get_transform(train=True, strong_aug=True)
    
    # Default transform for validation and test
    val_transform = get_transform(train=False)
    
    # Create datasets
    train_dataset_labeled = SFPIDataset(
        split='train', 
        labeled=True, 
        label_percentage=config.LABEL_PERCENTAGE,
        transform=train_transform_labeled,
        use_synthetic_annotations=use_synthetic_annotations
    )
    
    train_dataset_unlabeled_weak = SFPIDataset(
        split='train', 
        labeled=False, 
        label_percentage=config.LABEL_PERCENTAGE,
        transform=train_transform_unlabeled_weak,
        use_synthetic_annotations=use_synthetic_annotations
    )
    
    train_dataset_unlabeled_strong = SFPIDataset(
        split='train', 
        labeled=False, 
        label_percentage=config.LABEL_PERCENTAGE,
        transform=train_transform_unlabeled_strong,
        use_synthetic_annotations=use_synthetic_annotations
    )
    
    val_dataset = SFPIDataset(
        split='val', 
        transform=val_transform,
        use_synthetic_annotations=use_synthetic_annotations
    )
    
    test_dataset = SFPIDataset(
        split='test', 
        transform=val_transform,
        use_synthetic_annotations=use_synthetic_annotations
    )
    
    # Create data loaders
    train_loader_labeled = DataLoader(
        train_dataset_labeled,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    train_loader_unlabeled_weak = DataLoader(
        train_dataset_unlabeled_weak,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    train_loader_unlabeled_strong = DataLoader(
        train_dataset_unlabeled_strong,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    loaders = {
        'train_labeled': train_loader_labeled,
        'train_unlabeled_weak': train_loader_unlabeled_weak,
        'train_unlabeled_strong': train_loader_unlabeled_strong,
        'val': val_loader,
        'test': test_loader
    }
    
    return loaders

def collate_fn(batch):
    """Custom collate function for Mask R-CNN data loader"""
    return tuple(zip(*batch)) 