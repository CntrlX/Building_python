import os

class Config:
    """Configuration for training the Mask R-CNN model on SFPI dataset"""
    
    # Name of the experiment
    NAME = "SFPI_MaskRCNN"
    
    # Root directory of the project
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the dataset
    DATASET_DIR = os.path.join(ROOT_DIR, "data", "dataset", "SFPI", "SFPI")
    
    # Path to the images
    IMAGES_DIR = os.path.join(DATASET_DIR, "Images")
    
    # Path to the annotations
    ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "Annotations")
    
    # Furniture classes
    CLASSES = [
        "background", "armchair", "bed", "door1", "door2", 
        "sink1", "sink2", "sink3", "sink4", "sofa1", 
        "sofa2", "table1", "table2", "table3", "tub", 
        "window1", "window2"
    ]
    
    # Number of classes (including background)
    NUM_CLASSES = len(CLASSES)
    
    # Training settings
    BATCH_SIZE = 8
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # Semi-supervised settings
    USE_SEMI_SUPERVISED = True
    LABEL_PERCENTAGE = 1  # 1%, 5%, or 10%
    DATA_SAMPLING_RATIO = 0.2
    FOREGROUND_THRESHOLD = 0.9
    BOX_REGRESSION_THRESHOLD = 0.02
    NJITTER = 10
    
    # Number of iterations
    MAX_ITERATIONS = 80000
    LEARNING_RATE_STEPS = [30000, 40000]
    
    # Teacher-Student model parameters
    TEACHER_MOMENTUM = 0.999  # EMA weight for teacher model
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Mask R-CNN settings
    BACKBONE = "resnet101"  # or "resnet50"
    FPN_ON = True
    ROI_POSITIVE_RATIO = 0.33
    
    # Detection settings
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    
    # RPN settings
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROI settings
    ROI_POSITIVE_THRESHOLD = 0.5
    ROI_NEGATIVE_THRESHOLD = 0.5
    
    # GPU settings
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8  # Adjust based on your GPU memory 
    
    # Dataset loading
    NUM_WORKERS = 2  # Number of worker threads for data loading
    
    # Additional attributes
    # ... (keep the existing attributes)
    
    # ... (keep the existing methods)
    
    # ... (keep the existing code) 