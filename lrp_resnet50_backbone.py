"""
LRP ResNet50 Backbone (No Top)

A reusable ResNet50 backbone without the final classification layers,
designed for use in Siamese networks and other architectures that need
feature embeddings with LRP capability.
"""

import tensorflow as tf
from typing import List, Optional, Dict, Any
from clean_lrp_resnet50 import (
    LRPConv2D,
    LRPDense,
    LRPPassThrough,
    LRPGlobalAveragePooling2D,
    LRPMaxPooling2D,
    LRPZeroPadding2D,
    LRPBottleneck,
)


class LRPResNet50_NoTop(tf.keras.layers.Layer):
    """
    LRP-capable ResNet50 backbone without final classification layers.

    This class wraps a pre-trained ResNet50 model from a Siamese network
    that was created without pooling or top layers. It provides LRP capability
    for the existing trained model.

    Args:
        pretrained_model: Pre-trained ResNet50 model (from Siamese network)
        **kwargs: Additional keyword arguments
    """

    def __init__(self, pretrained_model: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)

        # Use the provided pre-trained model
        self.base_model = pretrained_model

        # Build LRP structure
        self.lrp_layers = {}
        self.lrp_bottlenecks = {}
        self.layer_activations = {}
        self._build_lrp_backbone()

    def _build_lrp_backbone(self):
        """Build LRP version of ResNet50 backbone"""
        print("Building LRP ResNet50 Backbone (No Top)...")

        # Get layer dictionary from base model
        layer_dict = {layer.name: layer for layer in self.base_model.layers}

        # Wrap initial layers
        initial_layer_names = [
            "conv1_pad",
            "conv1_conv",
            "conv1_bn",
            "conv1_relu",
            "pool1_pad",
            "pool1_pool",
        ]

        for name in initial_layer_names:
            if name in layer_dict:
                self.lrp_layers[name] = self._wrap_layer(
                    layer_dict[name]
                )  # Create bottleneck blocks
        bottleneck_info = self._get_bottleneck_info()
        for block_info in bottleneck_info:
            self.lrp_bottlenecks[block_info["name"]] = self._create_lrp_bottleneck(
                block_info, layer_dict
            )

        print(
            f"Built backbone with {len(self.lrp_layers)} individual layers and {len(self.lrp_bottlenecks)} bottlenecks"
        )

    def _wrap_layer(self, layer):
        """Wrap layer with appropriate LRP class"""
        if isinstance(layer, tf.keras.layers.Conv2D):
            return LRPConv2D(layer)
        elif isinstance(layer, tf.keras.layers.Dense):
            return LRPDense(layer)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            return LRPPassThrough(layer)
        elif isinstance(layer, (tf.keras.layers.ReLU, tf.keras.layers.Activation)):
            return LRPPassThrough(layer)
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return LRPGlobalAveragePooling2D(layer)
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            return LRPMaxPooling2D(layer)
        elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
            return LRPZeroPadding2D(layer)
        else:
            return LRPPassThrough(layer)

    def _get_bottleneck_info(self):
        """Get ResNet50 bottleneck block information"""
        blocks = []

        stages = [
            {"prefix": "conv2_block", "blocks": 3, "first_has_shortcut": True},
            {"prefix": "conv3_block", "blocks": 4, "first_has_shortcut": True},
            {"prefix": "conv4_block", "blocks": 6, "first_has_shortcut": True},
            {"prefix": "conv5_block", "blocks": 3, "first_has_shortcut": True},
        ]

        for stage in stages:
            for block_idx in range(1, stage["blocks"] + 1):
                block_name = f"{stage['prefix']}{block_idx}"
                has_shortcut = block_idx == 1 and stage["first_has_shortcut"]
                blocks.append({"name": block_name, "has_shortcut": has_shortcut})

        return blocks

    def _create_lrp_bottleneck(self, block_info, layer_dict):
        """Create LRP bottleneck from block info"""
        block_name = block_info["name"]

        # Main path layers
        main_layers = []
        for step in [1, 2, 3]:
            conv_name = f"{block_name}_{step}_conv"
            bn_name = f"{block_name}_{step}_bn"

            if conv_name in layer_dict:
                main_layers.append(layer_dict[conv_name])
            if bn_name in layer_dict:
                main_layers.append(layer_dict[bn_name])

            # Add ReLU after steps 1 and 2
            if step < 3:
                relu_name = f"{block_name}_{step}_relu"
                if relu_name in layer_dict:
                    main_layers.append(layer_dict[relu_name])

        # Shortcut layers (for downsample blocks)
        shortcut_layers = None
        if block_info["has_shortcut"]:
            conv_name = f"{block_name}_0_conv"
            bn_name = f"{block_name}_0_bn"

            if conv_name in layer_dict and bn_name in layer_dict:
                shortcut_layers = [layer_dict[conv_name], layer_dict[bn_name]]

        # Final ReLU
        final_relu_name = f"{block_name}_out"
        final_relu = layer_dict.get(final_relu_name)

        return LRPBottleneck(
            main_layers=main_layers,
            shortcut_layers=shortcut_layers,
            final_relu=final_relu,
            name=f"lrp_{block_name}",
        )

    def call(self, inputs, training=None):
        """Forward pass through backbone"""
        x = inputs
        self.layer_activations = {"input": inputs}

        # Initial layers
        initial_layer_names = [
            "conv1_pad",
            "conv1_conv",
            "conv1_bn",
            "conv1_relu",
            "pool1_pad",
            "pool1_pool",
        ]

        for name in initial_layer_names:
            if name in self.lrp_layers:
                x = self.lrp_layers[name](x, training=training)
                self.layer_activations[name] = x

        # Bottleneck blocks
        bottleneck_order = [
            "conv2_block1",
            "conv2_block2",
            "conv2_block3",
            "conv3_block1",
            "conv3_block2",
            "conv3_block3",
            "conv3_block4",
            "conv4_block1",
            "conv4_block2",
            "conv4_block3",
            "conv4_block4",
            "conv4_block5",
            "conv4_block6",
            "conv5_block1",
            "conv5_block2",
            "conv5_block3",
        ]

        for block_name in bottleneck_order:
            if block_name in self.lrp_bottlenecks:
                x = self.lrp_bottlenecks[block_name](x, training=training)
                self.layer_activations[block_name] = x

        return x

    def compute_relevance(
        self,
        relevance_input: tf.Tensor,
        rule: str = "z_plus",
        gamma: float = 0.0,
        debug: bool = False,
    ) -> tf.Tensor:
        """
        Compute LRP relevance propagation through the backbone

        Args:
            relevance_input: Relevance scores at the output of the backbone
            rule: LRP rule to use ("z_plus", "gamma", "z_b")
            gamma: Gamma parameter for gamma rule
            debug: Enable debug output

        Returns:
            Relevance scores at the input of the backbone
        """
        if not self.layer_activations:
            raise ValueError("No activations stored. Call forward pass first.")

        relevance = relevance_input

        if debug:
            print(
                f"Starting backbone relevance computation: {tf.reduce_sum(relevance).numpy():.6f}"
            )

        # Backprop through bottlenecks (reverse order)
        bottleneck_order = [
            "conv5_block3",
            "conv5_block2",
            "conv5_block1",
            "conv4_block6",
            "conv4_block5",
            "conv4_block4",
            "conv4_block3",
            "conv4_block2",
            "conv4_block1",
            "conv3_block4",
            "conv3_block3",
            "conv3_block2",
            "conv3_block1",
            "conv2_block3",
            "conv2_block2",
            "conv2_block1",
        ]

        for block_name in bottleneck_order:
            if block_name in self.lrp_bottlenecks:
                relevance = self.lrp_bottlenecks[block_name].compute_relevance(
                    relevance, rule=rule, gamma=gamma, debug=debug
                )
                if debug:
                    print(f"After {block_name}: {tf.reduce_sum(relevance).numpy():.6f}")

        # Backprop through initial layers
        initial_layers = [
            "pool1_pool",
            "pool1_pad",
            "conv1_relu",
            "conv1_bn",
            "conv1_conv",
            "conv1_pad",
        ]
        for layer_name in initial_layers:
            if layer_name in self.lrp_layers:
                layer_rule = "z_b" if layer_name == "conv1_conv" else rule
                relevance = self.lrp_layers[layer_name].compute_relevance(
                    relevance, rule=layer_rule, gamma=gamma, debug=debug
                )
                if debug:
                    print(f"After {layer_name}: {tf.reduce_sum(relevance).numpy():.6f}")

        return relevance

    def get_embedding_size(self) -> int:
        """Get the size of the output embedding"""
        # Get the output shape from the last bottleneck
        # ResNet50 without pooling outputs (batch, H, W, 2048)
        return self.base_model.output_shape[-1]

    def get_feature_maps(self, layer_name: str) -> Optional[tf.Tensor]:
        """Get feature maps from a specific layer"""
        return self.layer_activations.get(layer_name)


# Example usage and testing
def demo_backbone():
    """Demo of the LRP ResNet50 backbone"""
    print("=" * 50)
    print("LRP ResNet50 Backbone Demo")
    print("=" * 50)

    # Create a ResNet50 model similar to what would be used in Siamese network
    pretrained_resnet = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,  # Exclude final classification layers
        input_shape=(224, 224, 3),
    )

    # Create backbone with the pretrained model
    backbone = LRPResNet50_NoTop(pretrained_resnet)

    # Create dummy input
    dummy_input = tf.random.normal((2, 224, 224, 3))

    # Forward pass
    features = backbone(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output feature shape: {features.shape}")
    print(f"Feature depth: {backbone.get_embedding_size()}")

    # Test relevance computation
    dummy_relevance = tf.ones_like(features)
    input_relevance = backbone.compute_relevance(dummy_relevance, debug=True)
    print(f"Input relevance shape: {input_relevance.shape}")

    return backbone


if __name__ == "__main__":
    demo_backbone()
