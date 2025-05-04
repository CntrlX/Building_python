import os
import sys
import argparse
import torch
import traceback
import datetime
import json
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import configuration
from odm.configs.config import Config
from odm.utils.dataset import SFPIDataset, get_transform
from odm.utils.model import build_model
from odm.utils.trainer import SemiSupervisedTrainer

# Configure logging
def setup_logger(log_dir, name='main'):
    """Set up logger with file and console handlers"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_file = os.path.join(log_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def collate_fn(batch):
    """Custom collate function for data loader"""
    return tuple(zip(*batch))

def create_data_loaders(config, label_percentage, use_synthetic, batch_size=None):
    """Create data loaders for training and validation"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Create transforms
    transform = get_transform(train=True)
    transform_val = get_transform(train=False)
    
    # Create datasets
    try:
        # Split dataset based on label percentage
        dataset_train_labeled = SFPIDataset(
            split='train',
            labeled=True,
            label_percentage=label_percentage,
            transform=transform,
            use_synthetic_annotations=use_synthetic
        )
        
        dataset_train_unlabeled = SFPIDataset(
            split='train',
            labeled=False,
            label_percentage=label_percentage,
            transform=transform,
            use_synthetic_annotations=use_synthetic
        )
        
        # Create strongly augmented version of unlabeled dataset
        dataset_train_unlabeled_strong = SFPIDataset(
            split='train',
            labeled=False,
            label_percentage=label_percentage,
            transform=get_transform(train=True, strong_aug=True),
            use_synthetic_annotations=use_synthetic
        )
        
        # Create validation dataset
        dataset_val = SFPIDataset(
            split='val',
            labeled=True,
            label_percentage=1.0,
            transform=transform_val,
            use_synthetic_annotations=use_synthetic
        )
        
        # Log dataset sizes
        print(f"Labeled dataset size: {len(dataset_train_labeled)}")
        print(f"Unlabeled dataset size: {len(dataset_train_unlabeled)}")
        print(f"Validation dataset size: {len(dataset_val)}")
        
        # Create data loaders
        data_loader_train_labeled = torch.utils.data.DataLoader(
            dataset_train_labeled, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn
        )
        
        data_loader_train_unlabeled_weak = torch.utils.data.DataLoader(
            dataset_train_unlabeled, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn
        )
        
        data_loader_train_unlabeled_strong = torch.utils.data.DataLoader(
            dataset_train_unlabeled_strong, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn
        )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=1,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn
        )
        
        # Return data loaders
        data_loaders = {
            'train_labeled': data_loader_train_labeled,
            'train_unlabeled_weak': data_loader_train_unlabeled_weak,
            'train_unlabeled_strong': data_loader_train_unlabeled_strong,
            'val': data_loader_val
        }
        
        return data_loaders
    
    except Exception as e:
        print(f"Error creating data loaders: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return None

def train(config, args):
    """Train the model"""
    try:
        print("Setting up training...")
        print(f"Using device: {args.device}")
        print(f"Label percentage: {args.label_percentage}")
        print(f"Use synthetic annotations: {args.use_synthetic}")
        
        # Create data loaders
        data_loaders = create_data_loaders(
            config, 
            args.label_percentage, 
            args.use_synthetic,
            batch_size=args.batch_size
        )
        
        if data_loaders is None:
            print("Failed to create data loaders. Aborting training.")
            return
        
        # Create model
        model = build_model(
            config, 
            device=args.device,
            pretrained=True,
            resume=args.resume
        )
        
        # Create optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            model.student.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.MAX_ITERATIONS
        )
        
        # Create tensorboard writer
        log_dir = os.path.join(config.ROOT_DIR, "logs", "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)
        
        # Save training parameters
        params = {
            "device": args.device,
            "label_percentage": args.label_percentage,
            "batch_size": args.batch_size,
            "use_synthetic": args.use_synthetic,
            "learning_rate": config.LEARNING_RATE,
            "max_iterations": config.MAX_ITERATIONS,
            "resume": args.resume,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(log_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
        
        # Create trainer
        trainer = SemiSupervisedTrainer(
            model=model,
            data_loaders=data_loaders,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            config=config,
            writer=writer
        )
        
        # Train the model
        print("Starting training...")
        best_map = trainer.train()
        print(f"Training completed. Best mAP: {best_map:.4f}")
        
        return best_map
    
    except Exception as e:
        print(f"Error during training: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return None

def validate(config, args):
    """Validate the model"""
    try:
        print("Setting up validation...")
        print(f"Using device: {args.device}")
        
        # Create validation dataset
        transform_val = get_transform(train=False)
        dataset_val = SFPIDataset(
            split='val',
            labeled=True,
            label_percentage=1.0,
            transform=transform_val,
            use_synthetic_annotations=args.use_synthetic
        )
        
        # Create validation data loader
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=1,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn
        )
        
        # Create model
        model = build_model(
            config, 
            device=args.device,
            pretrained=False
        )
        
        # Load model weights
        if args.model_path:
            model_path = args.model_path
        else:
            # Find the latest model in the output directory
            model_dir = os.path.join(config.ROOT_DIR, "output", "models")
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if not model_files:
                print("No model files found.")
                return
            
            latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            model_path = os.path.join(model_dir, latest_model)
        
        print(f"Loading model from {model_path}")
        
        # Load model state
        model_state = torch.load(model_path, map_location=args.device)
        if 'student' in model_state:
            model.student.load_state_dict(model_state['student'])
        else:
            model.student.load_state_dict(model_state)
        
        # Create dummy data loaders
        data_loaders = {
            'val': data_loader_val
        }
        
        # Create dummy optimizer and scheduler
        optimizer = torch.optim.SGD(model.student.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
        
        # Create tensorboard writer
        log_dir = os.path.join(config.ROOT_DIR, "logs", "validation", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=log_dir)
        
        # Create trainer
        trainer = SemiSupervisedTrainer(
            model=model,
            data_loaders=data_loaders,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            config=config,
            writer=writer
        )
        
        # Evaluate the model
        print("Starting validation...")
        mAP = trainer.evaluate()
        print(f"Validation mAP: {mAP:.4f}")
        
        return mAP
    
    except Exception as e:
        print(f"Error during validation: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return None

def test_model(config, args):
    """Test the model on an image"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import torchvision.transforms as T
        import numpy as np
        
        print("Setting up model for testing...")
        print(f"Using device: {args.device}")
        
        # Create model
        model = build_model(
            config, 
            device=args.device,
            pretrained=False
        )
        
        # Load model weights
        if args.model_path:
            model_path = args.model_path
        else:
            # Find the best model in the output directory
            model_dir = os.path.join(config.ROOT_DIR, "output", "models")
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            
            if os.path.exists(best_model_path):
                model_path = best_model_path
            else:
                # Find the latest model
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if not model_files:
                    print("No model files found.")
                    return
                
                latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
                model_path = os.path.join(model_dir, latest_model)
        
        print(f"Loading model from {model_path}")
        
        # Load model state
        model_state = torch.load(model_path, map_location=args.device)
        if 'student' in model_state:
            model.student.load_state_dict(model_state['student'])
        else:
            model.student.load_state_dict(model_state)
        
        # Set model to evaluation mode
        model.student.eval()
        
        # Load image
        if not args.image_path:
            print("No image path provided. Please specify an image path with --image-path.")
            return
        
        image_path = args.image_path
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return
        
        print(f"Testing model on image: {image_path}")
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        transform = get_transform(train=False)
        
        img_tensor = transform(image).unsqueeze(0).to(args.device)
        
        # Run inference
        with torch.no_grad():
            prediction = model.student(img_tensor)[0]
        
        # Process predictions
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        pred_masks = None
        if 'masks' in prediction:
            pred_masks = prediction['masks'].cpu().numpy()
        
        # Filter predictions by confidence
        confidence_threshold = 0.5
        indices = pred_scores >= confidence_threshold
        pred_boxes = pred_boxes[indices]
        pred_scores = pred_scores[indices]
        pred_labels = pred_labels[indices]
        if pred_masks is not None:
            pred_masks = pred_masks[indices]
        
        # Set up output directory
        output_dir = os.path.join(config.ROOT_DIR, "output", "test_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure for visualization
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        
        # Draw predictions
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            label_text = f"Class {pred_labels[i]}: {pred_scores[i]:.2f}"
            ax.text(x1, y1-5, label_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # Display number of detections
        ax.set_title(f"Detections: {len(pred_boxes)}")
        
        # Save figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"test_result_{timestamp}.png")
        plt.savefig(output_path)
        print(f"Result saved to {output_path}")
        
        # Save detection data
        results = {
            "image_path": image_path,
            "num_detections": len(pred_boxes),
            "boxes": pred_boxes.tolist(),
            "scores": pred_scores.tolist(),
            "labels": pred_labels.tolist()
        }
        
        results_path = os.path.join(output_dir, f"test_result_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Detection data saved to {results_path}")
        
        # Try to display the image if running in interactive mode
        if sys.stdout.isatty():
            try:
                plt.show()
            except:
                pass
        
        return results
    
    except Exception as e:
        print(f"Error during testing: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Object Detection Model (ODM)")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "val", "test"], 
                        help="Mode: train, val, or test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (default: cuda if available, else cpu)")
    parser.add_argument("--label-percentage", type=float, default=1.0,
                        help="Percentage of labeled data to use for training (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (default: from config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic annotations when real annotations are missing")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model weights for validation or testing")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to image for testing")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set up logging
    config = Config()
    log_dir = os.path.join(config.ROOT_DIR, "logs")
    logger = setup_logger(log_dir)
    
    # Print start message
    logger.info("Starting ODM...")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    
    # Check if device is available
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Run based on mode
    try:
        if args.mode == "train":
            logger.info("Starting training...")
            train(config, args)
        elif args.mode == "val":
            logger.info("Starting validation...")
            validate(config, args)
        elif args.mode == "test":
            logger.info("Starting testing...")
            test_model(config, args)
    except Exception as e:
        logger.error(f"Error in main: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("Done.")

if __name__ == "__main__":
    main() 