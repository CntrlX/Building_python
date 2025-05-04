# Mask-Aware Semi-Supervised Object Detection

A PyTorch implementation of a semi-supervised object detection model inspired by research on semi-supervised learning approaches for object detection with masks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Validation](#validation)
- [Testing](#testing)
- [Synthetic Annotations](#synthetic-annotations)
- [Troubleshooting](#troubleshooting)
- [Directory Structure](#directory-structure)
- [Recent Improvements](#recent-improvements)

## Introduction

This project implements a semi-supervised object detection model that leverages both labeled and unlabeled data. The model uses a teacher-student framework where the teacher model generates pseudo-labels for unlabeled data, which are then used to train the student model along with labeled data.

The model is primarily designed for the SFPI dataset but can be adapted to other datasets as well.

## Features

- Semi-supervised learning for object detection
- Teacher-student model architecture with EMA updates
- Support for different label percentages (train with limited labeled data)
- Synthetic annotation generation for missing annotations
- Robust error handling and recovery
- Comprehensive logging and monitoring
- Support for CPU and CUDA training
- TensorBoard integration for tracking metrics

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the dataset:
   - Place images in the `data/images` directory
   - Place annotations in the `data/annotations` directory
   - Ensure the directory structure matches the configuration

## Usage

The project provides a convenient batch script for Windows users, and a Python module that can be run directly for all platforms.

### Using the Batch Script (Windows)

For Windows users, use the `train.bat` script with various options:

```
train.bat [options]
```

Options:
- `--mode [train|val|test]`: Mode to run (default: train)
- `--device [cuda|cpu]`: Device to use (default: cuda)
- `--label-percentage VALUE`: Percentage of labeled data to use (default: 1.0)
- `--batch-size VALUE`: Batch size for training
- `--resume`: Resume training from last checkpoint
- `--use-synthetic`: Use synthetic annotations when annotations are missing
- `--model-path PATH`: Path to model for validation/testing
- `--image-path PATH`: Path to image for testing
- `--test`: Run synthetic annotation test before training
- `--debug`: Enable debug output

Examples:
```
train.bat --mode train --device cuda --label-percentage 0.1 --use-synthetic
train.bat --mode val --model-path output/models/best_model.pth
train.bat --mode test --image-path path/to/image.jpg
```

### Using Python Module (All Platforms)

For all platforms, you can run the Python module directly:

```
python -m odm.main [options]
```

Options are the same as for the batch script.

## Training

To train the model, use the following command:

```
train.bat --mode train --device cuda --label-percentage 0.1 --use-synthetic
```

or

```
python -m odm.main --mode train --device cuda --label-percentage 0.1 --use-synthetic
```

Parameters:
- `--label-percentage`: Control how much labeled data to use (0.0-1.0)
- `--batch-size`: Set batch size for training (default from config)
- `--device`: Choose training device ('cuda' or 'cpu')
- `--use-synthetic`: Use synthetic annotations when real ones are missing
- `--resume`: Resume training from the last checkpoint

Training automatically saves checkpoints and logs progress to TensorBoard. The best model based on validation mAP is saved separately.

## Validation

To validate the model, use:

```
train.bat --mode val --model-path output/models/best_model.pth
```

or

```
python -m odm.main --mode val --model-path output/models/best_model.pth
```

If `--model-path` is not provided, the script will use the latest model in the output directory.

## Testing

To test the model on a specific image, use:

```
train.bat --mode test --image-path path/to/image.jpg
```

or

```
python -m odm.main --mode test --image-path path/to/image.jpg
```

This will generate a visualization of detections and save it to the output directory.

## Synthetic Annotations

The model supports synthetic annotation generation for cases when real annotations are missing or insufficient. This is useful for:

1. Training with incomplete datasets
2. Testing model behavior with artificial data
3. Diagnosing dataset issues

To use synthetic annotations, add the `--use-synthetic` flag to your command:

```
train.bat --mode train --use-synthetic
```

To test the synthetic annotation generation, use:

```
train.bat --test
```

This will run a test script that validates the synthetic annotation generation process and visualizes some examples.

## Troubleshooting

For common issues and their solutions, see the [Troubleshooting Guide](docs/troubleshooting.md).

## Directory Structure

```
odm/
├── configs/             # Configuration files
├── data/                # Dataset directory
│   ├── images/          # Images
│   └── annotations/     # Annotation files
├── docs/                # Documentation
├── logs/                # Log files
│   ├── tensorboard/     # TensorBoard logs
│   └── validation/      # Validation logs
├── model/               # Model implementation
├── output/              # Output files
│   ├── models/          # Saved models
│   └── test_results/    # Testing visualizations
├── utils/               # Utility functions
│   ├── dataset.py       # Dataset handling
│   ├── trainer.py       # Training logic
│   └── model.py         # Model building
├── main.py              # Main entry point
├── test_synthetic.py    # Test script for synthetic annotations
├── train.bat            # Training batch script for Windows
└── README.md            # This file
```

## Recent Improvements

### Error Handling and Robustness

- Added comprehensive error handling throughout the codebase
- Improved recovery from common errors during training and validation
- Implemented detailed logging for easier debugging
- Added memory monitoring and management for GPU training

### Large Image Handling

- Automatic detection and resizing of large images
- Memory usage tracking and reporting
- Warnings for potentially problematic images

### Training Enhancements

- Added support for resuming training from checkpoints
- Improved handling of batches with no valid boxes
- Implemented dynamic batch skipping with fallback to synthetic boxes
- Better progress reporting and ETA calculation

### Validation Improvements

- Continued validation despite individual sample errors
- Detailed reporting of validation errors
- Memory-efficient validation process

### Synthetic Annotations

- Enhanced synthetic annotation generation
- Added test script for verifying synthetic annotations
- Improved integration with the training pipeline

### User Interface

- Comprehensive command-line interface with more options
- Enhanced batch script for Windows users
- Better help documentation and examples
- Improved logging and progress reporting