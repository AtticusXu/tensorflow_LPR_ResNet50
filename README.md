# Clean LRP ResNet50 Implementation

A clean, well-organized implementation of Layer-wise Relevance Propagation (LRP) for ResNet50 that focuses on clarity and correctness. This implementation is based on the paper "Layer-Wise Relevance Propagation with Conservation Property for ResNet" (ECCV 2024).

## Overview

This implementation provides a simplified but complete LRP framework for ResNet50 that includes:

- **Multiple LRP Rules**: Support for z+, gamma, and z_b rules
- **Proper Bottleneck Handling**: Ratio-based splitting for ResNet bottleneck blocks  
- **Clean Architecture**: Well-organized code with clear separation of concerns
- **Easy to Use**: Simple interface for image analysis with built-in visualization

## Key Features

### LRP Rules Implemented

1. **z+ Rule**: For intermediate layers with non-negative activations
2. **Gamma Rule**: Configurable rule with `w + γ * w+` weight modification
3. **z_b Rule**: For the first convolutional layer with known pixel bounds

### ResNet50 Components

- **Initial Layers**: conv1, batch norm, ReLU, max pooling with proper padding handling
- **Bottleneck Blocks**: All 16 bottleneck blocks with ratio-based splitting
- **Final Layers**: Global average pooling and predictions layer

### Advanced Features

- **Ratio-Based Splitting**: Proper relevance distribution at skip connections
- **Heat Quantization**: Optional quantization of attribution maps
- **Activation Storage**: Efficient forward pass with activation caching
- **Debug Support**: Detailed relevance tracking and visualization

## Installation

```bash
pip install tensorflow matplotlib numpy
```

## Quick Start

```python
from clean_lrp_resnet50 import CleanLRPResNet50, load_sample_image

# Create LRP-enabled ResNet50
model = CleanLRPResNet50(weights="imagenet")

# Load and analyze an image
image = load_sample_image("path/to/your/image.jpg")
results = model.analyze_image(image, show_plot=True)

print(f"Predicted class: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.4f}")
```

## Detailed Usage

### Basic Image Analysis

```python
import tensorflow as tf
from clean_lrp_resnet50 import CleanLRPResNet50

# Create model
model = CleanLRPResNet50(weights="imagenet")

# Load your image (should be RGB, any size)
image = tf.io.read_file("your_image.jpg")
image = tf.image.decode_image(image, channels=3)
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, tf.float32)

# Analyze with LRP
results = model.analyze_image(image, show_plot=True)
```

### Advanced LRP Computation

```python
# For more control over the LRP process
processed_image = tf.keras.applications.resnet50.preprocess_input(image)
attribution_map = model.compute_lrp(
    processed_image, 
    class_idx=285,  # Target specific class
    debug=True      # Enable detailed output
)
```

### Custom Visualization

```python
import matplotlib.pyplot as plt

# Create custom visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(results["display_image"])
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(results["attribution_map"], cmap="RdBu_r")
axes[1].set_title("Attribution")
axes[1].axis("off")

axes[2].imshow(results["display_image"])
axes[2].imshow(results["attribution_map"], cmap="RdBu_r", alpha=0.5)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.show()
```

## Architecture Details

### LRP Layer Classes

- **`LRPConv2D`**: Handles convolutional layers with multiple rule support
- **`LRPDense`**: Handles fully connected layers
- **`LRPBottleneck`**: Handles ResNet bottleneck blocks with ratio-based splitting
- **`LRPGlobalAveragePooling2D`**: Handles global average pooling
- **`LRPMaxPooling2D`**: Handles max pooling (uses average pooling for LRP)
- **`LRPZeroPadding2D`**: Handles zero padding (removes padding in reverse)
- **`LRPPassThrough`**: Pass-through for activations and batch normalization

### ResNet50 Structure

The implementation correctly handles the complete ResNet50 architecture:

```
Input (224×224×3)
├── Initial Block: conv1_pad → conv1_conv → conv1_bn → conv1_relu → pool1_pad → pool1_pool
├── Stage 1: conv2_block1, conv2_block2, conv2_block3
├── Stage 2: conv3_block1, conv3_block2, conv3_block3, conv3_block4  
├── Stage 3: conv4_block1, conv4_block2, conv4_block3, conv4_block4, conv4_block5, conv4_block6
├── Stage 4: conv5_block1, conv5_block2, conv5_block3
└── Final Block: avg_pool → predictions
```

## LRP Rules in Detail

### z_b Rule (First Layer)

Applied to `conv1_conv` layer to handle preprocessed input with known bounds:

```
R_i = Σ_j [(z_ij - l_i * w_ij^+ - h_i * w_ij^-) / Σ_i'(z_i'j - l_i' * w_i'j^+ - h_i' * w_i'j^-)] * R_j
```

Where:
- `l_i, h_i` are the lower and upper bounds for pixel `i`
- `w_ij^+, w_ij^-` are positive and negative parts of weights
- Bounds account for ResNet50 preprocessing (mean subtraction)

### z+ Rule (Intermediate Layers)

Applied to most convolutional and dense layers:

```
R_i = Σ_j [a_i * w_ij^+ / Σ_i'(a_i' * w_i'j^+)] * R_j
```

Only positive weights contribute to relevance propagation.

### Ratio-Based Splitting (Bottlenecks)

At each Add operation in bottleneck blocks:

```
R_main = R_total * |main_output| / (|main_output| + |shortcut_output|)
R_shortcut = R_total * |shortcut_output| / (|main_output| + |shortcut_output|)
```

## Performance and Memory

### Computational Complexity

- **Forward Pass**: Same as standard ResNet50
- **LRP Computation**: ~2-3x standard forward pass time
- **Memory Usage**: Stores activations for all layers during forward pass

### Optimization Tips

1. **Batch Processing**: Process images one at a time to reduce memory usage
2. **Debug Mode**: Disable debug output for faster computation
3. **Visualization**: Set `show_plot=False` when processing multiple images

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or process images individually
2. **Shape Mismatches**: Ensure input images are RGB and properly resized
3. **Missing Layers**: Some ResNet50 layer names may vary between TensorFlow versions

### Debug Output

Enable debug mode to track relevance conservation:

```python
attribution = model.compute_lrp(image, debug=True)
```

This will show relevance sums at each layer to verify conservation property.

## References

```bibtex
@article{otsuki2024layer,
    title={{Layer-Wise Relevance Propagation with Conservation Property for ResNet}},
    author={Seitaro Otsuki, Tsumugi Iida, F\'elix Doublet, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, Komei Sugiura},
    journal={arXiv preprint arXiv:2407.09115},
    year={2024},
}
```

## License

This implementation is provided for research and educational purposes. Please cite the original paper when using this code in your research.

## Contributing

Contributions are welcome! Please ensure that:

1. Code follows the existing style and organization
2. New features include appropriate documentation
3. Changes maintain compatibility with the existing API
4. Tests are provided for new functionality

## File Structure

```
├── clean_lrp_resnet50.py    # Main implementation
├── README.md                # This file
└── examples/                # Example usage scripts
    ├── basic_usage.py       # Basic image analysis example
    ├── batch_processing.py  # Process multiple images
    └── custom_rules.py      # Custom LRP rule examples
```

## Example Output

The implementation produces high-quality attribution maps that highlight the most relevant pixels for the model's prediction. The z_b rule for the first layer ensures that the attribution properly handles the preprocessed input space, while ratio-based splitting maintains relevance conservation through skip connections.

For questions or issues, please refer to the original paper or create an issue in the repository.
