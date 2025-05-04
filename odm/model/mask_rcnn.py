import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from odm.configs.config import Config

def build_mask_rcnn_model(num_classes):
    """
    Build a Mask R-CNN model with a ResNet-50 or ResNet-101 backbone
    
    Args:
        num_classes (int): Number of classes to predict
    
    Returns:
        Mask R-CNN model
    """
    config = Config()
    
    # Load a pre-trained model for classification and return only the features
    if config.BACKBONE == "resnet101":
        backbone = torchvision.models.resnet101(pretrained=True)
    else:  # Default to ResNet-50
        backbone = torchvision.models.resnet50(pretrained=True)
    
    # Create a Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, 
        min_size=800, 
        max_size=1333,
        rpn_anchor_generator=None,  # Use default anchors
        box_detections_per_img=100,
        box_nms_thresh=config.DETECTION_NMS_THRESHOLD
    )
    
    # Replace the BoxPredictor with a new one that has num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the MaskPredictor with a new one that has num_classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

class TeacherStudentMaskRCNN(torch.nn.Module):
    """
    Teacher-Student architecture for semi-supervised learning with Mask R-CNN
    """
    
    def __init__(self, num_classes):
        """
        Initialize the Teacher-Student model
        
        Args:
            num_classes (int): Number of classes in the dataset
        """
        super(TeacherStudentMaskRCNN, self).__init__()
        self.config = Config()
        
        # Build the student model (the one that will be updated during training)
        self.student = build_mask_rcnn_model(num_classes)
        
        # Build the teacher model (will be updated with EMA)
        self.teacher = build_mask_rcnn_model(num_classes)
        
        # Initialize teacher model with student weights
        self._copy_student_to_teacher()
        
        # Set teacher model to evaluation mode
        self.teacher.eval()
    
    def _copy_student_to_teacher(self):
        """Copy weights from student to teacher model"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.copy_(student_param.data)
    
    def update_teacher(self):
        """
        Update teacher model using exponential moving average (EMA)
        of student model parameters
        """
        with torch.no_grad():
            m = self.config.TEACHER_MOMENTUM
            for teacher_param, student_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_param.data.mul_(m).add_(student_param.data, alpha=1 - m)
    
    def forward(self, images, targets=None, teacher_eval=False):
        """
        Forward pass through the model
        
        Args:
            images (List[Tensor]): Images to be processed
            targets (List[Dict], optional): Ground-truth for the image
            teacher_eval (bool): Whether to run teacher model in evaluation mode
        
        Returns:
            Dict or List[Dict]: The output from the model (losses during training, detections during inference)
        """
        if teacher_eval:
            # If teacher_eval is True, run the teacher model in evaluation mode
            with torch.no_grad():
                return self.teacher(images)
        else:
            # Otherwise, run the student model
            return self.student(images, targets)
    
    def generate_pseudo_labels(self, unlabeled_images, threshold=0.9):
        """
        Generate pseudo-labels for unlabeled images using the teacher model
        
        Args:
            unlabeled_images (List[Tensor]): Unlabeled images
            threshold (float): Confidence threshold for pseudo-labels
        
        Returns:
            List[Dict]: Pseudo-labels with boxes, scores, labels, and masks
        """
        self.teacher.eval()
        with torch.no_grad():
            # Get predictions from teacher model
            predictions = self.teacher(unlabeled_images)
            
            # Filter predictions by confidence threshold
            filtered_predictions = []
            for pred in predictions:
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                masks = pred['masks']
                
                # Get indices of predictions above threshold
                keep_idxs = scores > threshold
                
                # Filter predictions
                filtered_pred = {
                    'boxes': boxes[keep_idxs],
                    'scores': scores[keep_idxs],
                    'labels': labels[keep_idxs],
                    'masks': masks[keep_idxs]
                }
                
                filtered_predictions.append(filtered_pred)
            
            return filtered_predictions
    
    def box_jitter(self, boxes, width, height, jitter_scale=0.06):
        """
        Apply jittering to boxes for regression variance estimation
        
        Args:
            boxes (Tensor): Boxes to jitter
            width (int): Image width
            height (int): Image height
            jitter_scale (float): Scale of jittering
        
        Returns:
            Tensor: Jittered boxes
        """
        if boxes.shape[0] == 0:
            return boxes
        
        # Calculate jitter amount based on box width and height
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        jitter_widths = widths * jitter_scale
        jitter_heights = heights * jitter_scale
        
        # Generate random jitter
        jitter_x = torch.randn(boxes.shape[0]) * jitter_widths
        jitter_y = torch.randn(boxes.shape[0]) * jitter_heights
        
        # Apply jitter to each coordinate
        jittered_boxes = boxes.clone()
        jittered_boxes[:, 0] = (boxes[:, 0] + jitter_x).clamp(min=0, max=width)
        jittered_boxes[:, 1] = (boxes[:, 1] + jitter_y).clamp(min=0, max=height)
        jittered_boxes[:, 2] = (boxes[:, 2] + jitter_x).clamp(min=0, max=width)
        jittered_boxes[:, 3] = (boxes[:, 3] + jitter_y).clamp(min=0, max=height)
        
        return jittered_boxes
    
    def compute_regression_variance(self, boxes, images, num_jitter=10):
        """
        Compute regression variance for boxes
        
        Args:
            boxes (List[Tensor]): List of boxes for each image
            images (List[Tensor]): List of images
            num_jitter (int): Number of jitter iterations
        
        Returns:
            List[Tensor]: Regression variance for each box
        """
        self.teacher.eval()
        all_variances = []
        
        with torch.no_grad():
            for image_idx, (box, image) in enumerate(zip(boxes, images)):
                if box.shape[0] == 0:
                    all_variances.append(torch.empty(0, 4))
                    continue
                
                # Get image dimensions
                _, height, width = image.shape
                
                # Create storage for jittered results
                all_refined_boxes = []
                
                # Generate and refine jittered boxes
                for _ in range(num_jitter):
                    # Jitter boxes
                    jittered_box = self.box_jitter(box, width, height)
                    
                    # Feed jittered boxes to teacher model
                    # Create dummy target with jittered boxes
                    dummy_target = [{
                        'boxes': jittered_box,
                        'labels': torch.ones((jittered_box.shape[0],), dtype=torch.int64),
                        # Add dummy values for other required fields
                        'image_id': torch.tensor([image_idx]),
                        'area': (jittered_box[:, 3] - jittered_box[:, 1]) * (jittered_box[:, 2] - jittered_box[:, 0]),
                        'iscrowd': torch.zeros((jittered_box.shape[0],), dtype=torch.int64)
                    }]
                    
                    # Run inference with jittered boxes
                    refined_preds = self.teacher([image])
                    refined_boxes = refined_preds[0]['boxes']
                    
                    # Only keep refined boxes if they exist
                    if refined_boxes.shape[0] > 0:
                        all_refined_boxes.append(refined_boxes)
                
                # Compute variance if we have refined boxes
                if all_refined_boxes:
                    # Stack all refined boxes
                    all_refined_boxes = torch.stack(all_refined_boxes)
                    
                    # Compute standard deviation across jittered predictions
                    box_std = torch.std(all_refined_boxes, dim=0)
                    
                    # Normalize by box dimensions
                    box_widths = box[:, 2] - box[:, 0]
                    box_heights = box[:, 3] - box[:, 1]
                    box_sizes = 0.5 * (box_widths + box_heights).unsqueeze(1).repeat(1, 4)
                    
                    # Normalize standard deviation
                    normalized_std = box_std / box_sizes
                    
                    all_variances.append(normalized_std)
                else:
                    # If no refined boxes, assign high variance
                    all_variances.append(torch.ones((box.shape[0], 4)) * 1.0)
        
        return all_variances 