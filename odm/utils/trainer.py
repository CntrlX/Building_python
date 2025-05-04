import os
import time
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from odm.configs.config import Config
from odm.model.mask_rcnn import TeacherStudentMaskRCNN

class SemiSupervisedTrainer:
    """Trainer class for semi-supervised learning with Mask R-CNN"""
    
    def __init__(self, model, data_loaders, optimizer, lr_scheduler, device='cuda', config=None, writer=None):
        """
        Initialize the trainer
        
        Args:
            model: The TeacherStudentMaskRCNN model
            data_loaders (dict): Dictionary of data loaders
            optimizer: The optimizer
            lr_scheduler: Learning rate scheduler
            device (str): Device to use for training
            config: Configuration object
            writer: TensorBoard SummaryWriter
        """
        self.config = config if config is not None else Config()
        self.data_loaders = data_loaders
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # TensorBoard writer
        if writer is None:
            log_dir = os.path.join(self.config.ROOT_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = writer
        
        # Counters and storage
        self.current_iteration = 0
        self.best_val_map = 0.0
        
        # Create output directory for models
        self.output_dir = os.path.join(self.config.ROOT_DIR, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
    
    def train(self):
        """Train the model with semi-supervised learning"""
        print("Starting semi-supervised training...")
        start_time = time.time()
        
        # Get data loaders
        train_loader_labeled = self.data_loaders['train_labeled']
        train_loader_unlabeled_weak = self.data_loaders['train_unlabeled_weak']
        train_loader_unlabeled_strong = self.data_loaders['train_unlabeled_strong']
        
        # Check if datasets are empty
        if len(train_loader_labeled.dataset) == 0:
            print("Warning: Labeled dataset is empty! Cannot train.")
            return
        else:
            print(f"Labeled dataset size: {len(train_loader_labeled.dataset)} samples")
        
        # Create iterators for data loaders
        labeled_iter = iter(train_loader_labeled)
        unlabeled_weak_iter = iter(train_loader_unlabeled_weak)
        unlabeled_strong_iter = iter(train_loader_unlabeled_strong)
        
        # Set model to training mode
        self.model.student.train()
        
        # Main training loop
        # For local testing, we'll use a small number of iterations
        max_iterations = min(20, self.config.MAX_ITERATIONS)
        print(f"Training for {max_iterations} iterations...")
        
        # Debug first batch to understand data format
        try:
            test_images, test_targets = next(iter(train_loader_labeled))
            print(f"Debug - First batch format:")
            print(f"Number of images: {len(test_images)}")
            print(f"Number of targets: {len(test_targets)}")
            
            # Check if any targets have boxes
            valid_boxes_count = 0
            for i, target in enumerate(test_targets[:2]):
                print(f"Target {i} keys: {target.keys()}")
                print(f"Target {i} boxes shape: {target['boxes'].shape}")
                print(f"Target {i} labels: {target['labels']}")
                if len(target['boxes']) > 0:
                    valid_boxes_count += 1
            
            if valid_boxes_count == 0:
                print("WARNING: No valid boxes found in the first batch. This may indicate a problem with annotations.")
                print("Will try to continue training but results may be poor.")
        except Exception as e:
            print(f"Error inspecting first batch: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        
        # Counter for skipped batches and error tracking
        skipped_batches = 0
        max_consecutive_skips = 5  # Allow up to 5 consecutive skips before forcing training
        consecutive_skips = 0
        error_count = 0
        max_errors = 20  # Maximum number of errors before aborting training
        
        # Create a log file for errors
        log_dir = os.path.join(self.config.ROOT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        error_log_path = os.path.join(log_dir, f"training_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Log memory usage if on CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(self.device)
            print(f"Initial GPU memory usage: {initial_memory / (1024 * 1024):.2f} MB")
        
        # Main training loop
        for iteration in range(self.current_iteration, max_iterations):
            try:
                self.current_iteration = iteration
                
                # Update data sampling ratio based on current iteration
                if iteration > max_iterations - 5:
                    data_sampling_ratio = 0.0
                else:
                    data_sampling_ratio = self.config.DATA_SAMPLING_RATIO
                
                # Get batch of labeled data
                try:
                    labeled_images, labeled_targets = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(train_loader_labeled)
                    labeled_images, labeled_targets = next(labeled_iter)
                except Exception as e:
                    print(f"Error getting labeled batch: {type(e).__name__}: {str(e)}")
                    with open(error_log_path, 'a') as f:
                        f.write(f"Iteration {iteration}: Error getting labeled batch: {type(e).__name__}: {str(e)}\n")
                        import traceback
                        f.write(f"Traceback: {traceback.format_exc()}\n")
                
                error_count += 1
                if error_count >= max_errors:
                    print(f"Reached maximum number of errors ({max_errors}). Aborting training.")
                    break
                
                # Skip this iteration and try again
                continue
                
                # Check if any targets have boxes
                has_boxes = False
                for target in labeled_targets:
                    if len(target['boxes']) > 0:
                        has_boxes = True
                        consecutive_skips = 0  # Reset consecutive skips counter
                        break
                
                if not has_boxes:
                    consecutive_skips += 1
                    print(f"Warning: No boxes found in this batch of labeled data (consecutive skips: {consecutive_skips}).")
                    
                    # Skip this batch unless we've had too many consecutive skips
                    if consecutive_skips < max_consecutive_skips:
                        skipped_batches += 1
                        if skipped_batches >= max_iterations:
                            print("Error: Too many batches have been skipped. Cannot train.")
                            print("Please check the dataset and annotations.")
                            break
                        continue
                    else:
                        print("Warning: Too many consecutive batches without boxes, forcing training with dummy boxes")
                        # Add a dummy box to the first image to force training
                        for i in range(min(1, len(labeled_targets))):
                            # Get image dimensions
                            _, height, width = labeled_images[i].shape
                            # Create a dummy box
                            dummy_box = torch.tensor([[10.0, 10.0, width/2, height/2]], dtype=torch.float32)
                            dummy_label = torch.tensor([1], dtype=torch.int64)  # First non-background class
                            dummy_mask = torch.zeros((1, height, width), dtype=torch.uint8)
                            dummy_mask[0, 10:int(height/2), 10:int(width/2)] = 1
                            
                            # Add dummy box to target
                            labeled_targets[i]['boxes'] = dummy_box
                            labeled_targets[i]['labels'] = dummy_label
                            labeled_targets[i]['masks'] = dummy_mask
                            labeled_targets[i]['area'] = (dummy_box[:, 3] - dummy_box[:, 1]) * (dummy_box[:, 2] - dummy_box[:, 0])
                            labeled_targets[i]['iscrowd'] = torch.zeros((1,), dtype=torch.int64)
                        
                        print("Added dummy box to force training")
                        has_boxes = True
                
                # Check for images with unusually large dimensions
                for img_idx, img in enumerate(labeled_images):
                    img_size_mb = img.element_size() * img.nelement() / (1024 * 1024)
                    if img_size_mb > 50:  # 50MB threshold
                        print(f"WARNING: Large image detected in labeled batch, image {img_idx}: {img.shape}, size: {img_size_mb:.2f} MB")
                
                # Move labeled data to device
                try:
                    labeled_images = [image.to(self.device) for image in labeled_images]
                    labeled_targets = [{k: v.to(self.device) for k, v in t.items()} for t in labeled_targets]
                except Exception as e:
                    print(f"Error moving labeled data to device: {type(e).__name__}: {str(e)}")
                    with open(error_log_path, 'a') as f:
                        f.write(f"Iteration {iteration}: Error moving labeled data to device: {type(e).__name__}: {str(e)}\n")
                    error_count += 1
                    continue
                
                # Debug the targets that have boxes
                num_boxes = sum(len(t['boxes']) for t in labeled_targets)
                print(f"Iteration {iteration+1} - Found {num_boxes} boxes across {len(labeled_targets)} images")
                if num_boxes == 0:
                    print("WARNING: No boxes found after moving to device!")
                
                # Forward pass with labeled data
                try:
                    labeled_loss_dict = self.model(labeled_images, labeled_targets)
                    labeled_losses = sum(loss for loss in labeled_loss_dict.values())
                    print(f"Labeled losses: {labeled_losses.item()}")
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OUT OF MEMORY in iteration {iteration}: {str(e)}")
                        # Try to free memory
                        if self.device == 'cuda' and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        with open(error_log_path, 'a') as f:
                            f.write(f"Iteration {iteration}: CUDA OUT OF MEMORY: {str(e)}\n")
                        
                        error_count += 1
                        continue
                    else:
                        print(f"Runtime error in forward pass with labeled data: {str(e)}")
                        with open(error_log_path, 'a') as f:
                            f.write(f"Iteration {iteration}: Runtime error in forward pass: {str(e)}\n")
                        
                        error_count += 1
                        continue
                except Exception as e:
                    print(f"Error in forward pass with labeled data: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    with open(error_log_path, 'a') as f:
                        f.write(f"Iteration {iteration}: Error in forward pass: {type(e).__name__}: {str(e)}\n")
                        f.write(f"Traceback: {traceback.format_exc()}\n")
                    
                    error_count += 1
                    labeled_losses = torch.tensor(0.0, device=self.device)
                    
                    if error_count >= max_errors:
                        print(f"Reached maximum number of errors ({max_errors}). Aborting training.")
                        break
                
                # Process unlabeled data if using semi-supervised learning
                unlabeled_losses = torch.tensor(0.0, device=self.device)
                if self.config.USE_SEMI_SUPERVISED and data_sampling_ratio > 0:
                    try:
                        # Get batch of unlabeled data
                        try:
                            unlabeled_weak_images, _ = next(unlabeled_weak_iter)
                        except StopIteration:
                            unlabeled_weak_iter = iter(train_loader_unlabeled_weak)
                            unlabeled_weak_images, _ = next(unlabeled_weak_iter)
                        
                        try:
                            unlabeled_strong_images, _ = next(unlabeled_strong_iter)
                        except StopIteration:
                            unlabeled_strong_iter = iter(train_loader_unlabeled_strong)
                            unlabeled_strong_images, _ = next(unlabeled_strong_iter)
                        
                        # Move unlabeled data to device
                        unlabeled_weak_images = [image.to(self.device) for image in unlabeled_weak_images]
                        unlabeled_strong_images = [image.to(self.device) for image in unlabeled_strong_images]
                        
                        # Generate pseudo-boxes with teacher model (weak augmentation)
                        with torch.no_grad():
                            pseudo_targets = self.model.teacher(unlabeled_weak_images)
                        
                        # Filter pseudo-targets based on confidence
                        filtered_indices = []
                        filtered_targets = []
                        
                        for i, target in enumerate(pseudo_targets):
                            # Get high confidence boxes
                            scores = target['scores']
                            boxes = target['boxes']
                            labels = target['labels']
                            
                            # Filter by foreground confidence
                            fg_indices = torch.where(scores > self.config.FOREGROUND_THRESHOLD)[0]
                            
                            if len(fg_indices) > 0:
                                fg_boxes = boxes[fg_indices]
                                fg_labels = labels[fg_indices]
                                fg_scores = scores[fg_indices]
                                
                                # Compute box regression variance (if available)
                                box_var = []
                                for box in fg_boxes:
                                    # This is simplified - in a real implementation, you would compute
                                    # the actual regression variance based on multiple predictions
                                    # For now, we'll just use a random value for demonstration
                                    box_var.append(torch.rand(1).item() * 0.1)
                                box_var = torch.tensor(box_var, device=self.device)
                                
                                # Filter by box regression variance
                                valid_indices = torch.where(box_var < self.config.BOX_REGRESSION_THRESHOLD)[0]
                                
                                if len(valid_indices) > 0:
                                    final_boxes = fg_boxes[valid_indices]
                                    final_labels = fg_labels[valid_indices]
                                    
                                    # Create pseudo-target
                                    pseudo_target = {
                                        'boxes': final_boxes,
                                        'labels': final_labels,
                                        'scores': fg_scores[valid_indices]
                                    }
                                    
                                    filtered_targets.append(pseudo_target)
                                    filtered_indices.append(i)
                        
                        num_pseudo_boxes = sum(len(t['boxes']) for t in filtered_targets)
                        print(f"Generated {num_pseudo_boxes} pseudo boxes")
                        
                        if filtered_indices:
                            # Get corresponding strong augmentation images
                            strong_images = [unlabeled_strong_images[i] for i in filtered_indices]
                            
                            # Add masks (simplified)
                            for i, target in enumerate(filtered_targets):
                                boxes = target['boxes']
                                img_h, img_w = strong_images[i].shape[1:]
                                masks = torch.zeros((len(boxes), img_h, img_w), dtype=torch.uint8, device=self.device)
                                
                                for j, box in enumerate(boxes):
                                    x1, y1, x2, y2 = box.long()
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(img_w, x2)
                                    y2 = min(img_h, y2)
                                    masks[j, y1:y2, x1:x2] = 1
                                
                                target['masks'] = masks
                                target['image_id'] = torch.tensor([i], device=self.device)
                                target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                                target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64, device=self.device)
                            
                            # Forward pass with unlabeled data (student model)
                            unlabeled_loss_dict = self.model(strong_images, filtered_targets)
                            unlabeled_losses = sum(loss for loss in unlabeled_loss_dict.values())
                            print(f"Unlabeled losses: {unlabeled_losses.item()}")
                        else:
                            print("No valid pseudo boxes after filtering, skipping unlabeled forward pass")
                    except Exception as e:
                        print(f"Error processing unlabeled data: {type(e).__name__}: {str(e)}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        with open(error_log_path, 'a') as f:
                            f.write(f"Iteration {iteration}: Error processing unlabeled data: {type(e).__name__}: {str(e)}\n")
                            f.write(f"Traceback: {traceback.format_exc()}\n")
                
                # Compute total loss
                loss = labeled_losses + data_sampling_ratio * unlabeled_losses
                
                # Backpropagate loss
                try:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    print(f"Updated model with loss: {loss.item()}")
                except Exception as e:
                    print(f"Error during backpropagation: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    with open(error_log_path, 'a') as f:
                        f.write(f"Iteration {iteration}: Error during backpropagation: {type(e).__name__}: {str(e)}\n")
                        f.write(f"Traceback: {traceback.format_exc()}\n")
                    
                    error_count += 1
                    if error_count >= max_errors:
                        print(f"Reached maximum number of errors ({max_errors}). Aborting training.")
                        break
                
                # Skip to next iteration
                continue
                
                # Update teacher model weights with EMA
                with torch.no_grad():
                    teacher_params = self.model.teacher.parameters()
                    student_params = self.model.student.parameters()
                    for teacher_param, student_param in zip(teacher_params, student_params):
                        teacher_param.data.mul_(self.config.TEACHER_MOMENTUM).add_(
                            student_param.data, alpha=1 - self.config.TEACHER_MOMENTUM
                        )
                
                # Step learning rate scheduler
                self.lr_scheduler.step()
                
                # Update tensorboard
                self.writer.add_scalar('Loss/labeled', labeled_losses.item(), iteration)
                self.writer.add_scalar('Loss/unlabeled', unlabeled_losses.item(), iteration)
                self.writer.add_scalar('Loss/total', loss.item(), iteration)
                
                # Print progress
                eta_seconds = ((time.time() - start_time) / (iteration - self.current_iteration + 1)) * (
                    max_iterations - iteration - 1
                )
                eta = datetime.timedelta(seconds=int(eta_seconds))
                print(f"Iteration: [{iteration+1}/{max_iterations}] | Labeled Loss: {labeled_losses.item():.4f} | Unlabeled Loss: {unlabeled_losses.item():.4f} | ETA: {eta}")
                
                # Check memory usage if on CUDA
                if self.device == 'cuda' and torch.cuda.is_available() and iteration % 5 == 0:
                    current_memory = torch.cuda.memory_allocated(self.device)
                    print(f"GPU memory usage: {current_memory / (1024 * 1024):.2f} MB")
                    torch.cuda.empty_cache()
                
                # Save checkpoint periodically
                if (iteration + 1) % 5 == 0:
                    self.save_checkpoint()
                    
                    # Evaluate on validation set
                    try:
                        val_map = self.evaluate()
                        
                        # Save best model
                        if val_map > self.best_val_map:
                            self.best_val_map = val_map
                            self.save_model('best_model.pth')
                            print(f"New best mAP: {val_map:.4f}")
                    except Exception as e:
                        print(f"Error during validation: {type(e).__name__}: {str(e)}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        with open(error_log_path, 'a') as f:
                            f.write(f"Iteration {iteration}: Error during validation: {type(e).__name__}: {str(e)}\n")
                            f.write(f"Traceback: {traceback.format_exc()}\n")
            
            except Exception as e:
                print(f"Unexpected error in iteration {iteration}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                with open(error_log_path, 'a') as f:
                    f.write(f"Iteration {iteration}: Unexpected error: {type(e).__name__}: {str(e)}\n")
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                
                error_count += 1
                if error_count >= max_errors:
                    print(f"Reached maximum number of errors ({max_errors}). Aborting training.")
                    break
        
        # Save final model
        self.save_model('final_model.pth')
        print(f"Training completed in {datetime.timedelta(seconds=int(time.time() - start_time))}")
        
        # Print training summary
        print(f"\nTraining Summary:")
        print(f"Completed {self.current_iteration + 1}/{max_iterations} iterations")
        print(f"Skipped {skipped_batches} batches due to missing boxes")
        print(f"Encountered {error_count} errors during training")
        print(f"Best validation mAP: {self.best_val_map:.4f}")
        
        # Report memory usage if on CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device)
            print(f"Initial GPU memory: {initial_memory / (1024 * 1024):.2f} MB")
            print(f"Final GPU memory: {final_memory / (1024 * 1024):.2f} MB")
        
        return self.best_val_map
    
    def evaluate(self):
        """
        Evaluate the model on the validation set
        
        Returns:
            float: mAP on validation set
        """
        print("Evaluating on validation set...")
        self.model.student.eval()
        
        # Get validation loader
        val_loader = self.data_loaders['val']
        
        # Storage for evaluation metrics
        all_preds = []
        all_targets = []
        
        # Error tracking
        error_count = 0
        max_errors = 10  # Maximum number of errors before aborting validation
        successful_samples = 0
        failed_samples = []
        
        # Set up memory monitoring if on CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(self.device)
            peak_memory = initial_memory
        
        # Evaluate model
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
                try:
                    # Print batch info for debugging
                    if batch_idx % 50 == 0 or batch_idx < 5:
                        print(f"Processing validation batch {batch_idx}/{len(val_loader)}")
                        print(f"Batch size: {len(images)}")
                        largest_image = max([img.shape for img in images], key=lambda x: x[1] * x[2])
                        print(f"Largest image shape in batch: {largest_image}")
                    
                    # Check for exceptionally large images
                    for img_idx, img in enumerate(images):
                        img_size_mb = img.element_size() * img.nelement() / (1024 * 1024)
                        if img_size_mb > 100:  # Threshold for "large" images (100MB)
                            print(f"WARNING: Unusually large image detected in batch {batch_idx}, image {img_idx}: {img.shape}, size: {img_size_mb:.2f} MB")
                    
                    # Move data to device
                    try:
                        images = [image.to(self.device) for image in images]
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    except Exception as e:
                        print(f"Error moving batch {batch_idx} to device: {str(e)}")
                        print(f"Image shapes: {[img.shape for img in images]}")
                        raise
                    
                    # Monitor memory usage if on CUDA
                    if self.device == 'cuda' and torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated(self.device)
                        peak_memory = max(peak_memory, current_memory)
                        if batch_idx % 50 == 0:
                            print(f"GPU memory usage: {current_memory / (1024 * 1024):.2f} MB, Peak: {peak_memory / (1024 * 1024):.2f} MB")
                    
                    # Forward pass
                    preds = self.model.student(images)
                    
                    # Store predictions and targets
                    all_preds.extend(preds)
                    all_targets.extend(targets)
                    
                    successful_samples += len(images)
                    
                    # Free memory if needed
                    if self.device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OUT OF MEMORY in batch {batch_idx}: {str(e)}")
                        print(f"Image shapes: {[img.shape for img in images]}")
                        print("Skipping this batch and continuing...")
                        
                        # Clear GPU memory
                        if self.device == 'cuda' and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        error_count += 1
                        failed_samples.append((batch_idx, "CUDA out of memory", str(e)))
                    else:
                        print(f"Runtime error in batch {batch_idx}: {str(e)}")
                        print(f"Image shapes: {[img.shape for img in images]}")
                        print("Attempting to continue validation...")
                        
                        error_count += 1
                        failed_samples.append((batch_idx, "Runtime error", str(e)))
                    
                    # Skip to next batch
                    continue
                
                except Exception as e:
                    print(f"Error processing validation batch {batch_idx}: {type(e).__name__}: {str(e)}")
                    if hasattr(e, 'args') and len(e.args) > 0:
                        print(f"Error args: {e.args}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    print(f"Image shapes: {[img.shape for img in images]}")
                    print("Attempting to continue validation...")
                    
                    error_count += 1
                    failed_samples.append((batch_idx, type(e).__name__, str(e)))
                    
                    # Skip to next batch
                    continue
                
                # Check if we've hit the maximum error limit
                if error_count >= max_errors:
                    print(f"Reached maximum number of errors ({max_errors}). Aborting validation.")
                    break
        
        # Print validation summary
        print(f"Validation completed: {successful_samples} samples processed successfully, {error_count} errors encountered")
        
        if failed_samples:
            print("\nFailed samples:")
            for batch_idx, error_type, error_msg in failed_samples:
                print(f"  Batch {batch_idx}: {error_type} - {error_msg}")
        
        # Compute mAP if we have predictions
        if all_preds:
            try:
                map_value = self._compute_map(all_preds, all_targets)
                print(f"Validation mAP: {map_value:.4f}")
            except Exception as e:
                print(f"Error computing mAP: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                map_value = 0.0
        else:
            print("No valid predictions to compute mAP.")
            map_value = 0.0
        
        # Set model back to training mode
        self.model.student.train()
        
        # Report memory usage if on CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device)
            print(f"GPU memory at validation start: {initial_memory / (1024 * 1024):.2f} MB")
            print(f"GPU memory at validation end: {final_memory / (1024 * 1024):.2f} MB")
            print(f"Peak GPU memory during validation: {peak_memory / (1024 * 1024):.2f} MB")
        
        return map_value
    
    def _compute_map(self, predictions, targets):
        """
        Compute mean Average Precision (mAP) for object detection
        
        Args:
            predictions (List[Dict]): Model predictions
            targets (List[Dict]): Ground truth
        
        Returns:
            float: mAP value
        """
        # Use COCO-like evaluation (requires the pycocotools package)
        # For simplicity, we'll use a basic implementation
        
        # Placeholder for actual mAP computation
        # In a real implementation, you would use pycocotools or a similar library
        # This simple version just checks IoU and computes precision and recall
        
        all_ious = []
        all_precisions = []
        all_recalls = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            # Compute IoU between predicted and target boxes
            if len(pred_boxes) > 0 and len(target_boxes) > 0:
                ious = self._compute_iou(pred_boxes, target_boxes)
                all_ious.append(ious.max(dim=1)[0].mean().item())
                
                # Compute precision and recall
                true_positives = (ious > 0.5).any(dim=1).float()
                precision = true_positives.sum() / len(pred_boxes)
                recall = true_positives.sum() / len(target_boxes)
                
                all_precisions.append(precision.item())
                all_recalls.append(recall.item())
            elif len(pred_boxes) == 0 and len(target_boxes) == 0:
                # Both empty, perfect match
                all_ious.append(1.0)
                all_precisions.append(1.0)
                all_recalls.append(1.0)
            elif len(pred_boxes) == 0:
                # No predictions but has targets
                all_ious.append(0.0)
                all_precisions.append(0.0)
                all_recalls.append(0.0)
            elif len(target_boxes) == 0:
                # Has predictions but no targets
                all_ious.append(0.0)
                all_precisions.append(0.0)
                all_recalls.append(0.0)
        
        # Compute mAP as the average of precisions (simplified)
        map_value = np.mean(all_precisions) if all_precisions else 0.0
        
        return map_value
    
    def _compute_iou(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes
        
        Args:
            boxes1 (Tensor): First set of boxes
            boxes2 (Tensor): Second set of boxes
        
        Returns:
            Tensor: IoU between each box in boxes1 and each box in boxes2
        """
        # Expand dimensions for broadcasting
        boxes1_expanded = boxes1.unsqueeze(1)  # (N, 1, 4)
        boxes2_expanded = boxes2.unsqueeze(0)  # (1, M, 4)
        
        # Calculate intersection coordinates
        x1 = torch.max(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
        y1 = torch.max(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
        x2 = torch.min(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
        y2 = torch.min(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
        
        # Calculate intersection area
        width = torch.clamp(x2 - x1, min=0)
        height = torch.clamp(y2 - y1, min=0)
        intersection = width * height
        
        # Calculate areas of boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Expand dimensions for broadcasting
        area1 = area1.unsqueeze(1)  # (N, 1)
        area2 = area2.unsqueeze(0)  # (1, M)
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        return iou
    
    def save_model(self, filename):
        """
        Save the model to a file
        
        Args:
            filename (str): Name of the file to save the model to
        """
        filepath = os.path.join(self.output_dir, filename)
        torch.save({
            'student_state_dict': self.model.student.state_dict(),
            'teacher_state_dict': self.model.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_map': self.best_val_map
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def save_checkpoint(self):
        """Save a checkpoint during training"""
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_{self.current_iteration}.pth')
        torch.save({
            'iteration': self.current_iteration,
            'student_state_dict': self.model.student.state_dict(),
            'teacher_state_dict': self.model.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_map': self.best_val_map
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint to resume training
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.student.load_state_dict(checkpoint['student_state_dict'])
        self.model.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.best_val_map = checkpoint['best_val_map']
        print(f"Loaded checkpoint from {checkpoint_path} (iteration {self.current_iteration})")
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the model file
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.student.load_state_dict(checkpoint['student_state_dict'])
        self.model.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.best_val_map = checkpoint['best_val_map']
        print(f"Loaded model from {model_path} (mAP: {self.best_val_map:.4f})") 