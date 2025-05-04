import torch
import torchvision
from odm.model.mask_rcnn import TeacherStudentMaskRCNN
from odm.configs.config import Config

def build_model(config, device='cuda', pretrained=True, resume=False):
    """
    Build the model for training or evaluation
    
    Args:
        config: Configuration object
        device: Device to place the model on
        pretrained: Whether to use pretrained weights
        resume: Whether to resume training from a checkpoint
    
    Returns:
        TeacherStudentMaskRCNN model
    """
    print(f"Building model on {device}...")
    
    # Create the model
    model = TeacherStudentMaskRCNN(
        num_classes=config.NUM_CLASSES
    )
    
    # Move the model to the device
    model.to(device)
    
    # Load model weights if resuming training
    if resume:
        try:
            # Find the latest checkpoint
            import os
            import glob
            
            model_dir = os.path.join(config.ROOT_DIR, "output", "models")
            checkpoints = glob.glob(os.path.join(model_dir, "*.pth"))
            
            if not checkpoints:
                print("No checkpoints found. Starting from scratch.")
                return model
            
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Loading checkpoint: {latest_checkpoint}")
            
            # Load the checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            
            # Load model state
            if 'student' in checkpoint:
                model.student.load_state_dict(checkpoint['student'])
                model.teacher.load_state_dict(checkpoint['teacher'])
            else:
                model.student.load_state_dict(checkpoint)
                model.teacher.load_state_dict(checkpoint)
            
            print("Checkpoint loaded successfully.")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch.")
    
    return model 