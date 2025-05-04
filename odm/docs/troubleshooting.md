# Troubleshooting Guide: Semi-Supervised Object Detection Model

This guide provides solutions for common issues encountered when training, validating, or testing the Mask-Aware Semi-Supervised Object Detection Model.

## Table of Contents

1. [Data Loading Issues](#data-loading-issues)
2. [Memory Issues](#memory-issues)
3. [Training Issues](#training-issues)
4. [Validation Issues](#validation-issues)
5. [Synthetic Annotation Issues](#synthetic-annotation-issues)
6. [CUDA Issues](#cuda-issues)
7. [Batch Size Issues](#batch-size-issues)
8. [Understanding Logs](#understanding-logs)
9. [Common Error Messages](#common-error-messages)

## Data Loading Issues

### Missing Annotation Files

**Symptoms:**
- Warnings about missing annotation files
- "No valid boxes found" messages during training

**Solutions:**
- Use synthetic annotations by adding the `--use-synthetic` flag
- Check that your dataset directory structure is correct
- Verify that annotation files match the expected format
- Run the test script to verify dataset loading: `train.bat --test`

### Image Loading Errors

**Symptoms:**
- Errors related to image loading during training or validation
- Messages about corrupt or invalid images

**Solutions:**
- The system now automatically resizes large images (>4000px) to prevent issues
- Check for corrupted images in your dataset
- Verify image format compatibility (supported formats: jpg, png, tif, tiff)
- Inspect the error logs in the `logs` directory for specific image filenames

### Dataset Size Issues

**Symptoms:**
- Warnings about empty datasets
- Very small number of samples reported

**Solutions:**
- Verify your dataset path in the configuration
- Check that the dataset split ('train', 'val') exists
- Ensure label percentage is appropriate (e.g., try 1.0 for using all labeled data)
- Run with debug flag to see more information: `train.bat --debug`

## Memory Issues

### CUDA Out of Memory

**Symptoms:**
- "CUDA out of memory" errors during training
- Training fails after processing a few batches

**Solutions:**
- Reduce batch size using the `--batch-size` flag (e.g., `--batch-size 2`)
- Use a smaller image size by modifying the transforms in the dataset
- Try CPU training with `--device cpu` (slower but uses system memory)
- Check for unusually large images in your dataset (now automatically logged)
- The system now automatically logs memory usage and tries to recover from OOM errors

### Large Image Handling

**Symptoms:**
- Warnings about large images detected
- Memory spikes during training

**Solutions:**
- The system now automatically resizes images larger than 4000px in any dimension
- Consider preprocessing your dataset to normalize image sizes
- Monitor memory usage in the logs
- Reduce batch size if you have many large images

## Training Issues

### No Valid Boxes Found

**Symptoms:**
- Warnings about no valid boxes found in batches
- Skipping multiple batches during training

**Solutions:**
- The system will now add dummy boxes after multiple consecutive skips
- Use synthetic annotations with `--use-synthetic`
- Check your annotation files for issues
- Verify the class IDs in annotations match your configuration
- Increase label percentage to use more labeled data

### Training Stalls or Makes No Progress

**Symptoms:**
- Loss doesn't decrease
- Training seems to make no progress

**Solutions:**
- Check the loss values in the logs
- Verify that there are enough valid boxes in your dataset
- Try with synthetic annotations as a baseline
- Check that your learning rate is appropriate
- Resume training from a checkpoint with `--resume`

### Invalid Loss Values

**Symptoms:**
- NaN or Inf loss values reported
- Training crashes after a few iterations

**Solutions:**
- Check for extremely small or large values in your annotations
- Verify that your model is properly initialized
- Reduced batch size to improve stability
- Check the logs for specific error messages about invalid computations

## Validation Issues

### Validation Fails

**Symptoms:**
- Errors during validation
- Validation stops before processing all samples

**Solutions:**
- The system now continues validation even if individual samples fail
- Check the validation logs for specific errors
- Run validation separately: `train.bat --mode val`
- Use a specific model for validation: `train.bat --mode val --model-path PATH`

### Low mAP Results

**Symptoms:**
- Very low mAP values reported after validation
- Detection results don't match expectations

**Solutions:**
- Verify your ground truth annotations
- Try with more labeled data (higher label percentage)
- Check that class IDs are consistent between training and validation
- Inspect visualizations of test results: `train.bat --mode test --image-path PATH`

## Synthetic Annotation Issues

### Synthetic Annotations Not Working

**Symptoms:**
- Still getting "no boxes found" despite using `--use-synthetic`
- Poor training results with synthetic annotations

**Solutions:**
- Run the synthetic annotation test: `train.bat --test`
- Check that synthetic annotations are properly generated (logs will show numbers)
- Try adjusting the number of synthetic annotations per image in the dataset code
- Use `--debug` flag to see more detailed information

### Inconsistent Results with Synthetic Annotations

**Symptoms:**
- Different results each time you train with synthetic annotations
- Unpredictable model behavior

**Solutions:**
- This is expected as synthetic annotations are randomly generated
- For reproducibility, you can set a fixed random seed in the dataset code
- Synthetic annotations are meant as a fallback, not a primary training method
- Try to improve your real annotations whenever possible

## CUDA Issues

### CUDA Not Available

**Symptoms:**
- Warnings about CUDA not being available
- Falling back to CPU despite specifying `--device cuda`

**Solutions:**
- Verify that your GPU supports CUDA
- Check that CUDA and PyTorch are properly installed
- Update GPU drivers
- Check GPU utilization with Task Manager or `nvidia-smi`

### CUDA Version Mismatch

**Symptoms:**
- Errors about CUDA version compatibility
- PyTorch cannot find CUDA

**Solutions:**
- Ensure PyTorch is installed with the correct CUDA version
- Check compatibility between your GPU, CUDA version, and PyTorch version
- Consider using a compatible Docker container

## Batch Size Issues

### Optimal Batch Size

**Symptoms:**
- Training is too slow or unstable
- Uncertain what batch size to use

**Solutions:**
- Start with a small batch size (2 or 4) and increase if memory allows
- Monitor memory usage (automatically logged every 5 iterations)
- Larger batch sizes usually give better results but require more memory
- For low-memory GPUs, try batch size 1 or 2

### Batch Size and Learning Rate

**Symptoms:**
- Training diverges or stalls with large batch sizes
- Loss values are unstable

**Solutions:**
- When increasing batch size, consider adjusting learning rate
- A common rule: if you multiply batch size by N, multiply learning rate by √N
- The system uses AdamW optimizer which can handle various batch sizes well

## Understanding Logs

### Log Files

The system creates several types of log files:

- **Main logs**: Located in `/logs/main_*.log`
- **Training error logs**: Located in `/logs/training_errors_*.log`
- **TensorBoard logs**: Located in `/logs/tensorboard/*/`
- **Validation logs**: Located in `/logs/validation/*/`

Use these logs to diagnose issues and track progress.

### Log Analysis

**Key information in logs:**
- Number of boxes found per batch
- Loss values (labeled, unlabeled, total)
- Memory usage statistics
- Validation mAP
- Errors encountered during training or validation

**Identifying issues from logs:**
- Consecutive skipped batches indicates annotation problems
- Memory spikes suggest large images
- Repeated errors on specific samples may indicate corrupt data
- Look for patterns in when errors occur

## Common Error Messages

### "IndexError: index out of range"

**Possible causes:**
- Empty tensors due to filtering operations
- No boxes in annotations
- Incorrect indexing in custom code

**Solutions:**
- The system now adds safeguards for empty tensors
- Check annotation files for errors
- Run with synthetic annotations

### "CUDA out of memory"

**Possible causes:**
- Batch size too large
- Images too large
- Model too complex for available GPU memory

**Solutions:**
- Reduce batch size
- The system now automatically resizes large images
- Try using CPU if GPU memory is insufficient

### "No such file or directory"

**Possible causes:**
- Incorrect dataset path
- Missing annotation files
- Path formatting issues (Windows vs. Unix)

**Solutions:**
- Verify all paths in configuration
- Check that dataset exists and has correct structure
- Use os.path functions for path manipulation

### "RuntimeError: Assertion failed"

**Possible causes:**
- Tensor shape mismatches
- NaN values in computation
- CUDA synchronization issues

**Solutions:**
- Check input and output tensor shapes
- Look for NaN values in your data
- Try CPU training to eliminate CUDA-specific issues

## Additional Help

If you continue to experience issues after trying these solutions:

1. Check the full logs for detailed error messages
2. Update to the latest version of the codebase
3. Try with a simpler dataset to isolate issues
4. Run the test script to verify components individually
5. Consider reducing model complexity as a test 