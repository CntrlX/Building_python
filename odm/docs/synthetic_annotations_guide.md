# Synthetic Annotations Guide

This guide explains how synthetic annotations work in the Mask-Aware Semi-Supervised Object Detection model and how to configure them for your specific use case.

## Introduction

Synthetic annotations are automatically generated bounding boxes, class labels, and masks that serve as a replacement for real annotations when they are unavailable. They enable training with limited or no labeled data, which is particularly useful in semi-supervised learning scenarios.

## How Synthetic Annotations Work

When the system cannot find real annotations for an image, it generates synthetic annotations as follows:

1. **Box Generation**: Random bounding boxes are created with reasonable sizes (between 10% and 30% of the image dimensions)
2. **Label Assignment**: Each box is assigned a random class label from the available classes (excluding background)
3. **Mask Creation**: A simple rectangular mask is created for each box
4. **Target Format**: The annotations are converted to the target format expected by the model

## Configuration

You can enable or disable synthetic annotations when creating a dataset or data loaders:

```python
# Enable synthetic annotations
dataset = SFPIDataset(split='train', use_synthetic_annotations=True)

# Disable synthetic annotations
dataset = SFPIDataset(split='train', use_synthetic_annotations=False)

# When creating data loaders
data_loaders = get_data_loaders(config, use_synthetic_annotations=True)
```

You can also enable synthetic annotations from the command line when running the main script:

```bash
python -m odm.main --mode train --use-synthetic
```

## Customization

To customize synthetic annotation generation, you can modify the `_create_synthetic_annotations_for_image` and `_create_synthetic_annotations` methods in the `SFPIDataset` class in `odm/utils/dataset.py`.

### Customize Box Size

The default size range for boxes is 10% to 30% of the image dimensions. You can adjust this by modifying:

```python
box_width = random.randint(int(width * 0.1), int(width * 0.3))
box_height = random.randint(int(height * 0.1), int(height * 0.3))
```

### Customize Number of Boxes

The default number of boxes per image is between 3 and 6. You can adjust this by modifying:

```python
num_boxes = random.randint(3, 6)
```

### Customize Label Distribution

By default, class labels are assigned randomly. You can implement a more sophisticated distribution by modifying:

```python
label = random.randint(1, max_label)
```

For example, to bias towards certain classes:

```python
# Bias towards furniture classes (e.g., classes 3-8)
if random.random() < 0.7:  # 70% chance of furniture
    label = random.randint(3, 8)
else:  # 30% chance of other classes
    label = random.choice(list(range(1, 3)) + list(range(9, max_label + 1)))
```

## Best Practices

1. **Test First**: Always run the test script to ensure synthetic annotations are working correctly
2. **Visualization**: Check the output directory for visualizations of synthetic annotations
3. **Annotation Counts**: Monitor the number of annotations being generated for each image
4. **Label Balance**: Ensure a reasonable balance of classes in synthetic annotations
5. **Transition**: Use synthetic annotations as a stepping stone, and gradually incorporate more real annotations

## Troubleshooting

### No Boxes Found in Batch

If you see warnings about "No boxes found in batch", make sure:
- The synthetic annotation code is working correctly
- The box dimensions are valid (non-zero width and height)
- The number of boxes per image is sufficient

### Invalid Labels

If you see errors about invalid label indices ("Target X is out of bounds"), ensure:
- The maximum label index is within valid range (less than `config.NUM_CLASSES`)
- Labels start from 1 (not 0, which is reserved for background)

### Visualization Issues

If visualization shows incorrect boxes or masks:
- Check that box coordinates are within image boundaries
- Verify that masks are properly aligned with boxes
- Ensure the image dimensions are correctly handled

## Performance Considerations

Synthetic annotations are a good starting point but have limitations:
- They don't capture the true distribution of objects in real data
- They may not represent realistic spatial relationships between objects
- They provide less precise learning signals than real annotations

For best results, gradually incorporate more real annotations as they become available, while still leveraging the synthetic annotations for unlabeled data. 