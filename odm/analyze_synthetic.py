import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from collections import Counter, defaultdict
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odm.configs.config import Config
from odm.utils.dataset import SFPIDataset, get_data_loaders


def analyze_dataset_annotations(dataset, num_samples=100, output_dir=None):
    """
    Analyze annotations from a dataset
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to analyze
        output_dir: Directory to save figures
    """
    if output_dir is None:
        output_dir = os.path.join(Config().ROOT_DIR, "output", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics to collect
    box_counts = []
    box_sizes = []
    box_aspect_ratios = []
    label_counts = Counter()
    box_positions = []
    
    # Collect statistics from samples
    print(f"Analyzing {min(num_samples, len(dataset))} samples...")
    for i in range(min(num_samples, len(dataset))):
        try:
            _, target = dataset[i]
            
            # Box count per image
            boxes = target['boxes']
            num_boxes = len(boxes)
            box_counts.append(num_boxes)
            
            if num_boxes > 0:
                # Box sizes
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                areas = widths * heights
                box_sizes.extend(areas.numpy())
                
                # Box aspect ratios
                ratios = widths / heights
                box_aspect_ratios.extend(ratios.numpy())
                
                # Label distribution
                for label in target['labels'].numpy():
                    label_counts[label] += 1
                
                # Box positions (center coordinates normalized to 0-1)
                centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
                centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
                box_positions.extend(zip(centers_x.numpy(), centers_y.numpy()))
        except Exception as e:
            print(f"Error analyzing sample {i}: {e}")
    
    # Plot statistics
    
    # 1. Box count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(box_counts, kde=True, bins=max(10, max(box_counts) - min(box_counts) + 1))
    plt.title("Distribution of Box Counts per Image")
    plt.xlabel("Number of Boxes")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "box_count_distribution.png"))
    plt.close()
    
    # 2. Box size distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(box_sizes, kde=True, bins=30)
    plt.title("Distribution of Box Sizes")
    plt.xlabel("Box Area (pixels²)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "box_size_distribution.png"))
    plt.close()
    
    # 3. Box aspect ratio distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(box_aspect_ratios, kde=True, bins=30)
    plt.title("Distribution of Box Aspect Ratios")
    plt.xlabel("Width/Height Ratio")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "box_aspect_ratio_distribution.png"))
    plt.close()
    
    # 4. Label distribution
    plt.figure(figsize=(12, 8))
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    indices = np.arange(len(labels))
    plt.bar(indices, counts)
    plt.title("Distribution of Class Labels")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.xticks(indices, labels)
    plt.savefig(os.path.join(output_dir, "label_distribution.png"))
    plt.close()
    
    # 5. Box position heatmap
    plt.figure(figsize=(10, 8))
    x_pos, y_pos = zip(*box_positions) if box_positions else ([], [])
    plt.hexbin(x_pos, y_pos, gridsize=20, cmap='viridis')
    plt.title("Spatial Distribution of Boxes")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.colorbar(label="Count")
    plt.savefig(os.path.join(output_dir, "box_position_heatmap.png"))
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples analyzed: {min(num_samples, len(dataset))}")
    print(f"Total boxes: {sum(box_counts)}")
    print(f"Mean boxes per image: {np.mean(box_counts):.2f}")
    print(f"Box count range: {min(box_counts)}-{max(box_counts)}")
    print(f"Number of classes detected: {len(label_counts)}")
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        class_name = Config().CLASSES[label] if label < len(Config().CLASSES) else f"Unknown-{label}"
        print(f"  Class {label} ({class_name}): {count} boxes ({count/sum(label_counts.values())*100:.1f}%)")
    
    # Return statistics for further analysis
    return {
        'box_counts': box_counts,
        'box_sizes': box_sizes,
        'box_aspect_ratios': box_aspect_ratios,
        'label_counts': label_counts,
        'box_positions': box_positions
    }


def compare_real_and_synthetic(output_dir=None):
    """Compare real and synthetic annotations"""
    if output_dir is None:
        output_dir = os.path.join(Config().ROOT_DIR, "output", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    real_dataset = SFPIDataset(split='train', use_synthetic_annotations=False)
    synthetic_dataset = SFPIDataset(split='train', use_synthetic_annotations=True)
    
    print("Analyzing real annotations...")
    real_stats = analyze_dataset_annotations(real_dataset, output_dir=os.path.join(output_dir, "real"))
    
    print("\nAnalyzing synthetic annotations...")
    synthetic_stats = analyze_dataset_annotations(synthetic_dataset, output_dir=os.path.join(output_dir, "synthetic"))
    
    # Compare box counts
    plt.figure(figsize=(12, 8))
    sns.histplot(real_stats['box_counts'], kde=True, label="Real", alpha=0.5, bins=10)
    sns.histplot(synthetic_stats['box_counts'], kde=True, label="Synthetic", alpha=0.5, bins=10)
    plt.title("Comparison of Box Counts per Image")
    plt.xlabel("Number of Boxes")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "comparison_box_counts.png"))
    plt.close()
    
    # Compare label distributions
    real_labels = sorted(real_stats['label_counts'].keys())
    real_counts = [real_stats['label_counts'].get(label, 0) for label in real_labels]
    
    synthetic_labels = sorted(synthetic_stats['label_counts'].keys())
    synthetic_counts = [synthetic_stats['label_counts'].get(label, 0) for label in synthetic_labels]
    
    all_labels = sorted(set(real_labels) | set(synthetic_labels))
    
    real_counts_all = [real_stats['label_counts'].get(label, 0) for label in all_labels]
    synthetic_counts_all = [synthetic_stats['label_counts'].get(label, 0) for label in all_labels]
    
    # Convert to percentages
    total_real = sum(real_counts_all)
    total_synthetic = sum(synthetic_counts_all)
    
    real_pct = [count/total_real*100 if total_real > 0 else 0 for count in real_counts_all]
    synthetic_pct = [count/total_synthetic*100 if total_synthetic > 0 else 0 for count in synthetic_counts_all]
    
    # Plot
    plt.figure(figsize=(14, 8))
    width = 0.35
    indices = np.arange(len(all_labels))
    
    plt.bar(indices - width/2, real_pct, width, label="Real")
    plt.bar(indices + width/2, synthetic_pct, width, label="Synthetic")
    
    plt.title("Comparison of Class Label Distributions")
    plt.xlabel("Class Label")
    plt.ylabel("Percentage")
    plt.xticks(indices, all_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_label_distribution.png"))
    plt.close()
    
    print("\nComparison complete. Results saved to", output_dir)


def main():
    parser = argparse.ArgumentParser(description='Analyze synthetic annotations')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save output')
    parser.add_argument('--mode', type=str, default='both', choices=['real', 'synthetic', 'both', 'compare'],
                        help='Analysis mode')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(Config().ROOT_DIR, "output", "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'real':
        dataset = SFPIDataset(split='train', use_synthetic_annotations=False)
        analyze_dataset_annotations(dataset, args.num_samples, args.output_dir)
    elif args.mode == 'synthetic':
        dataset = SFPIDataset(split='train', use_synthetic_annotations=True)
        analyze_dataset_annotations(dataset, args.num_samples, args.output_dir)
    elif args.mode == 'both':
        real_dataset = SFPIDataset(split='train', use_synthetic_annotations=False)
        synthetic_dataset = SFPIDataset(split='train', use_synthetic_annotations=True)
        
        print("Analyzing real annotations...")
        analyze_dataset_annotations(real_dataset, args.num_samples, os.path.join(args.output_dir, "real"))
        
        print("\nAnalyzing synthetic annotations...")
        analyze_dataset_annotations(synthetic_dataset, args.num_samples, os.path.join(args.output_dir, "synthetic"))
    elif args.mode == 'compare':
        compare_real_and_synthetic(args.output_dir)


if __name__ == "__main__":
    main() 