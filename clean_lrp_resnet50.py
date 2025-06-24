"""
Clean LRP ResNet50 Implementation

A simplified, well-organized implementation of Layer-wise Relevance Propagation (LRP)
for ResNet50 that focuses on clarity and correctness.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Union
import warnings


class LRPLayer(tf.keras.layers.Layer):
    """Base class for LRP-capable layers"""

    def __init__(self, original_layer: tf.keras.layers.Layer, eps: float = 1e-6):
        super().__init__()
        self.original_layer = original_layer
        self.eps = eps
        self.input_activations = None

    def call(self, inputs, training=None):
        """Store activations and forward through original layer"""
        self.input_activations = inputs
        return self.original_layer(inputs, training=training)

    def compute_relevance(self, relevance_output: tf.Tensor, **kwargs) -> tf.Tensor:
        """Override in subclasses"""
        raise NotImplementedError


class LRPConv2D(LRPLayer):
    """LRP for Conv2D layers with support for z+, gamma, and z_b rules"""

    def compute_relevance(
        self,
        relevance_output: tf.Tensor,
        rule: str = "z_plus",
        gamma: float = 0.0,
        debug: bool = False,
    ) -> tf.Tensor:
        if self.input_activations is None:
            raise ValueError("No activations stored. Call forward pass first.")

        a = self.input_activations
        r = relevance_output
        kernel = self.original_layer.kernel
        bias = getattr(self.original_layer, "bias", None)

        if debug:
            print(f"    Conv2D {self.original_layer.name}: rule={rule}")
            print(
                f"      Input shape: {a.shape}, relevance sum: {tf.reduce_sum(r).numpy():.6f}"
            )

        if rule == "z_b":
            # z_b rule for input layer with box constraints
            return self._compute_z_b_rule(a, r, kernel, bias, debug)
        elif rule == "z_plus":
            # z+ rule (positive weights only)
            return self._compute_z_plus_rule(a, r, kernel, bias, debug)
        elif rule == "gamma":
            # Gamma rule
            return self._compute_gamma_rule(a, r, kernel, bias, gamma, debug)
        else:
            raise ValueError(f"Unknown rule: {rule}")

    def _compute_z_b_rule(self, a, r, kernel, bias, debug):
        """z_b rule for input layer with known pixel bounds"""
        # Define bounds for preprocessed ResNet50 input
        mean_bgr = [103.939, 116.779, 123.68]
        l_vals = [0.0 - m for m in mean_bgr]  # Lower bounds
        h_vals = [255.0 - m for m in mean_bgr]  # Upper bounds

        # Create bound tensors
        l = tf.constant(l_vals, dtype=a.dtype)
        l = tf.reshape(l, (1, 1, 1, a.shape[-1]))
        h = tf.constant(h_vals, dtype=a.dtype)
        h = tf.reshape(h, (1, 1, 1, a.shape[-1]))

        l_tensor = tf.ones_like(a) * l
        h_tensor = tf.ones_like(a) * h

        # Split weights
        w_pos = tf.maximum(0.0, kernel)
        w_neg = tf.minimum(0.0, kernel)

        def conv_forward(x, w, b):
            z = tf.nn.conv2d(
                x,
                w,
                strides=self.original_layer.strides,
                padding=self.original_layer.padding.upper(),
                dilations=self.original_layer.dilation_rate,
            )
            if b is not None:
                z = tf.nn.bias_add(z, b)
            return z

        # Compute modified denominator: z - l*w+ - h*w-
        z = conv_forward(a, kernel, bias)
        z_prime = (
            z
            - conv_forward(l_tensor, w_pos, None)
            - conv_forward(h_tensor, w_neg, None)
        )

        # Relevance ratios
        s = r / (z_prime + self.eps)

        # Gradient computation for each term
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(a)
            z_term = conv_forward(a, kernel, bias)
            z_pos_term = conv_forward(a, w_pos, None)
            z_neg_term = conv_forward(a, w_neg, None)

            out_main = tf.reduce_sum(z_term * tf.stop_gradient(s))
            out_pos = tf.reduce_sum(z_pos_term * tf.stop_gradient(s))
            out_neg = tf.reduce_sum(z_neg_term * tf.stop_gradient(s))

        c_main = tape.gradient(out_main, a)
        c_pos = tape.gradient(out_pos, a)
        c_neg = tape.gradient(out_neg, a)
        del tape

        if any(c is None for c in [c_main, c_pos, c_neg]):
            print(f"      WARNING: Gradient is None for z_b rule")
            return tf.zeros_like(a)

        # Combine terms: a*c - l*c_pos - h*c_neg
        r_new = a * c_main - l_tensor * c_pos - h_tensor * c_neg

        if debug:
            print(f"      Output relevance sum: {tf.reduce_sum(r_new).numpy():.6f}")

        return r_new

    def _compute_z_plus_rule(self, a, r, kernel, bias, debug):
        """z+ rule (positive weights only)"""
        w_modified = tf.maximum(0.0, kernel)
        bias_modified = tf.maximum(0.0, bias) if bias is not None else None

        return self._compute_with_modified_weights(
            a, r, w_modified, bias_modified, debug
        )

    def _compute_gamma_rule(self, a, r, kernel, bias, gamma, debug):
        """Gamma rule: w + gamma * w+"""
        w_pos = tf.maximum(0.0, kernel)
        w_modified = kernel + gamma * w_pos

        if bias is not None:
            bias_pos = tf.maximum(0.0, bias)
            bias_modified = bias + gamma * bias_pos
        else:
            bias_modified = None

        return self._compute_with_modified_weights(
            a, r, w_modified, bias_modified, debug
        )

    def _compute_with_modified_weights(self, a, r, w_modified, bias_modified, debug):
        """Common gradient computation with modified weights"""

        def forward_modified(x):
            z = tf.nn.conv2d(
                x,
                w_modified,
                strides=self.original_layer.strides,
                padding=self.original_layer.padding.upper(),
                dilations=self.original_layer.dilation_rate,
            )
            if bias_modified is not None:
                z = tf.nn.bias_add(z, bias_modified)
            return z

        with tf.GradientTape() as tape:
            tape.watch(a)
            z = forward_modified(a) + self.eps
            s = r / (z + 1e-9)
            output = tf.reduce_sum(z * tf.stop_gradient(s))

        c = tape.gradient(output, a)
        if c is None:
            print(f"      WARNING: Gradient is None")
            return tf.zeros_like(a)

        r_new = a * c

        if debug:
            print(f"      Output relevance sum: {tf.reduce_sum(r_new).numpy():.6f}")

        return r_new


class LRPDense(LRPLayer):
    """LRP for Dense layers"""

    def compute_relevance(
        self,
        relevance_output: tf.Tensor,
        rule: str = "z_plus",
        gamma: float = 0.0,
        debug: bool = False,
    ) -> tf.Tensor:
        if self.input_activations is None:
            raise ValueError("No activations stored")

        a = self.input_activations
        r = relevance_output
        kernel = self.original_layer.kernel
        bias = getattr(self.original_layer, "bias", None)

        if rule == "z_plus":
            w_modified = tf.maximum(0.0, kernel)
            bias_modified = tf.zeros_like(bias) if bias is not None else None
        elif rule == "gamma":
            w_pos = tf.maximum(0.0, kernel)
            w_modified = kernel + gamma * w_pos
            if bias is not None:
                bias_pos = tf.maximum(0.0, bias)
                bias_modified = bias + gamma * bias_pos
            else:
                bias_modified = None
        else:
            w_modified = kernel
            bias_modified = bias

        def forward_modified(x):
            z = tf.matmul(x, w_modified)
            if bias_modified is not None:
                z = tf.nn.bias_add(z, bias_modified)
            return z

        with tf.GradientTape() as tape:
            tape.watch(a)
            z = forward_modified(a) + self.eps
            s = r / (z + 1e-9)
            output = tf.reduce_sum(z * tf.stop_gradient(s))

        c = tape.gradient(output, a)
        if c is None:
            return tf.zeros_like(a)

        return a * c


class LRPPassThrough(LRPLayer):
    """Pass-through layer for activations, batch norm, etc."""

    def compute_relevance(self, relevance_output: tf.Tensor, **kwargs) -> tf.Tensor:
        return relevance_output


class LRPGlobalAveragePooling2D(LRPLayer):
    """LRP for Global Average Pooling"""

    def compute_relevance(self, relevance_output: tf.Tensor, **kwargs) -> tf.Tensor:
        if self.input_activations is None:
            raise ValueError("No activations stored")

        a = self.input_activations
        r = relevance_output

        # Handle shape mismatch
        if len(r.shape) == 4 and r.shape[1] == 1 and r.shape[2] == 1:
            r = tf.squeeze(r, axis=[1, 2])

        with tf.GradientTape() as tape:
            tape.watch(a)
            z = self.original_layer(a) + self.eps
            s = r / (z + 1e-9)
            output = tf.reduce_sum(z * tf.stop_gradient(s))

        c = tape.gradient(output, a)
        if c is None:
            return tf.zeros_like(a)

        return a * c


class LRPMaxPooling2D(LRPLayer):
    """LRP for Max Pooling (using average pooling for LRP)"""

    def compute_relevance(self, relevance_output: tf.Tensor, **kwargs) -> tf.Tensor:
        if self.input_activations is None:
            raise ValueError("No activations stored")

        a = self.input_activations
        r = relevance_output

        # Use average pooling for LRP
        avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=self.original_layer.pool_size,
            strides=self.original_layer.strides,
            padding=self.original_layer.padding,
        )

        with tf.GradientTape() as tape:
            tape.watch(a)
            z = avg_pool(a) + self.eps
            s = r / (z + 1e-9)
            output = tf.reduce_sum(z * tf.stop_gradient(s))

        c = tape.gradient(output, a)
        if c is None:
            return tf.zeros_like(a)

        return a * c


class LRPZeroPadding2D(LRPLayer):
    """LRP for Zero Padding (remove padding from relevance)"""

    def compute_relevance(self, relevance_output: tf.Tensor, **kwargs) -> tf.Tensor:
        if self.input_activations is None:
            raise ValueError("No activations stored")

        # Get padding configuration
        padding = self.original_layer.padding

        if isinstance(padding, int):
            pad_top = pad_bottom = pad_left = pad_right = padding
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            if isinstance(padding[0], (tuple, list)):
                pad_top, pad_bottom = padding[0]
                pad_left, pad_right = padding[1]
            else:
                pad_top = pad_bottom = padding[0]
                pad_left = pad_right = padding[1]
        else:
            return relevance_output

        # Remove padding
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            h_start = pad_top
            h_end = (
                relevance_output.shape[1] - pad_bottom
                if pad_bottom > 0
                else relevance_output.shape[1]
            )
            w_start = pad_left
            w_end = (
                relevance_output.shape[2] - pad_right
                if pad_right > 0
                else relevance_output.shape[2]
            )

            return relevance_output[:, h_start:h_end, w_start:w_end, :]

        return relevance_output


class LRPBottleneck(tf.keras.layers.Layer):
    """LRP for ResNet Bottleneck with ratio-based splitting"""

    def __init__(
        self,
        main_layers: List[tf.keras.layers.Layer],
        shortcut_layers: Optional[List[tf.keras.layers.Layer]] = None,
        final_relu: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Wrap layers for LRP
        self.main_lrp_layers = [self._wrap_layer(layer) for layer in main_layers]
        self.shortcut_lrp_layers = (
            [self._wrap_layer(layer) for layer in shortcut_layers]
            if shortcut_layers
            else None
        )
        self.final_relu_lrp = self._wrap_layer(final_relu) if final_relu else None

        # Store activations
        self.input_activations = None
        self.main_output = None
        self.shortcut_output = None
        self.add_output = None

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

    def call(self, inputs, training=None):
        """Forward pass through bottleneck"""
        self.input_activations = inputs

        # Main path
        x = inputs
        for layer in self.main_lrp_layers:
            x = layer(x, training=training)
        self.main_output = x

        # Shortcut path
        if self.shortcut_lrp_layers:
            shortcut = inputs
            for layer in self.shortcut_lrp_layers:
                shortcut = layer(shortcut, training=training)
            self.shortcut_output = shortcut
        else:
            self.shortcut_output = inputs

        # Add operation
        self.add_output = self.main_output + self.shortcut_output

        # Final ReLU
        if self.final_relu_lrp:
            output = self.final_relu_lrp(self.add_output, training=training)
        else:
            output = self.add_output

        return output

    def compute_relevance(
        self,
        relevance_output: tf.Tensor,
        rule: str = "z_plus",
        gamma: float = 0.0,
        debug: bool = False,
    ) -> tf.Tensor:
        """Compute relevance using ratio-based splitting"""
        if self.main_output is None or self.shortcut_output is None:
            raise ValueError("No activations stored")

        # Backprop through final ReLU
        if self.final_relu_lrp:
            r_after_add = self.final_relu_lrp.compute_relevance(
                relevance_output, rule=rule, gamma=gamma
            )
        else:
            r_after_add = relevance_output

        # Ratio-based splitting at Add operation
        main_abs = tf.abs(self.main_output)
        shortcut_abs = tf.abs(self.shortcut_output)
        denominator = main_abs + shortcut_abs + 1e-9

        r_main = r_after_add * (main_abs / denominator)
        r_shortcut = r_after_add * (shortcut_abs / denominator)

        if debug:
            print(f"    Bottleneck {self.name}:")
            print(f"      Main relevance: {tf.reduce_sum(r_main).numpy():.6f}")
            print(f"      Shortcut relevance: {tf.reduce_sum(r_shortcut).numpy():.6f}")

        # Backprop through main path
        r = r_main
        for layer in reversed(self.main_lrp_layers):
            r = layer.compute_relevance(r, rule=rule, gamma=gamma, debug=debug)
        main_relevance = r

        # Backprop through shortcut path
        if self.shortcut_lrp_layers:
            r = r_shortcut
            for layer in reversed(self.shortcut_lrp_layers):
                r = layer.compute_relevance(r, rule=rule, gamma=gamma, debug=debug)
            shortcut_relevance = r
        else:
            shortcut_relevance = r_shortcut

        return main_relevance + shortcut_relevance


class CleanLRPResNet50(tf.keras.Model):
    """Clean LRP implementation for ResNet50"""

    def __init__(self, weights="imagenet", **kwargs):
        super().__init__(**kwargs)

        # Load base model
        self.base_model = tf.keras.applications.ResNet50(
            weights=weights, include_top=True, input_shape=(224, 224, 3)
        )

        # Parse architecture and build LRP model
        self.lrp_layers = {}
        self.lrp_bottlenecks = {}
        self.layer_activations = {}
        self._build_lrp_model()

    def _build_lrp_model(self):
        """Build LRP version of the model"""
        print("Building Clean LRP ResNet50...")

        # Get layer dict
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
                self.lrp_layers[name] = self._wrap_layer(layer_dict[name])

        # Create bottleneck blocks
        bottleneck_info = self._get_bottleneck_info()
        for block_info in bottleneck_info:
            self.lrp_bottlenecks[block_info["name"]] = self._create_lrp_bottleneck(
                block_info, layer_dict
            )

        # Wrap final layers
        final_layer_names = ["avg_pool", "predictions"]
        for name in final_layer_names:
            if name in layer_dict:
                self.lrp_layers[name] = self._wrap_layer(layer_dict[name])

        print(
            f"Built {len(self.lrp_layers)} individual layers and {len(self.lrp_bottlenecks)} bottlenecks"
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
        """Get bottleneck block information"""
        blocks = []

        # Define ResNet50 bottleneck structure
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
        """Standard forward pass"""
        return self.base_model(inputs, training=training)

    def call_with_activations(self, inputs, training=None):
        """Forward pass that stores activations for LRP"""
        x = inputs
        self.layer_activations = {"input": inputs}

        # Initial layers
        for name in [
            "conv1_pad",
            "conv1_conv",
            "conv1_bn",
            "conv1_relu",
            "pool1_pad",
            "pool1_pool",
        ]:
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

        # Final layers
        for name in ["avg_pool", "predictions"]:
            if name in self.lrp_layers:
                x = self.lrp_layers[name](x, training=training)
                self.layer_activations[name] = x

        return x

    def compute_lrp(
        self, image: tf.Tensor, class_idx: Optional[int] = None, debug: bool = False
    ) -> tf.Tensor:
        """Compute LRP relevance scores"""
        print("\n=== Computing LRP ===")

        # Ensure batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)

        # Forward pass with activation storage
        predictions = self.call_with_activations(image, training=False)

        # Determine target class
        if class_idx is None:
            class_idx = tf.argmax(predictions[0]).numpy()
        confidence = predictions[0, class_idx].numpy()

        print(f"Target class: {class_idx}, Confidence: {confidence:.4f}")

        # Initialize relevance
        relevance = tf.zeros_like(predictions)
        relevance = tf.tensor_scatter_nd_update(relevance, [[0, class_idx]], [1.0])

        if debug:
            print(f"Initial relevance sum: {tf.reduce_sum(relevance).numpy():.6f}")

        # Backpropagate through predictions (LRP-0 rule)
        if "predictions" in self.lrp_layers:
            relevance = self.lrp_layers["predictions"].compute_relevance(
                relevance, rule="lrp_0", debug=debug
            )

        # Backpropagate through avg_pool
        if "avg_pool" in self.lrp_layers:
            relevance = self.lrp_layers["avg_pool"].compute_relevance(
                relevance, debug=debug
            )

        # Backpropagate through bottlenecks (reverse order)
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
                    relevance, rule="z_plus", debug=debug
                )
                if debug:
                    print(f"After {block_name}: {tf.reduce_sum(relevance).numpy():.6f}")

        # Backpropagate through initial layers
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
                rule = "z_b" if layer_name == "conv1_conv" else "z_plus"
                relevance = self.lrp_layers[layer_name].compute_relevance(
                    relevance, rule=rule, debug=debug
                )
                if debug:
                    print(f"After {layer_name}: {tf.reduce_sum(relevance).numpy():.6f}")

        # Sum over channels to get final attribution map
        if len(relevance.shape) == 4:
            attribution_map = tf.reduce_sum(relevance[0], axis=-1)
        else:
            attribution_map = relevance[0]

        print(f"Final attribution sum: {tf.reduce_sum(attribution_map).numpy():.6f}")

        return attribution_map

    def analyze_image(
        self, image: tf.Tensor, class_idx: Optional[int] = None, show_plot: bool = True
    ) -> Dict[str, Any]:
        """Complete image analysis with LRP"""
        # Preprocess image
        if tf.reduce_max(image) <= 1.0:
            display_image = image
            processed_image = image * 255.0
        else:
            display_image = image / 255.0
            processed_image = image

        # Apply ResNet50 preprocessing
        processed_image = tf.keras.applications.resnet50.preprocess_input(
            processed_image
        )

        # Get predictions
        predictions = self(tf.expand_dims(processed_image, 0))
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = tf.nn.softmax(predictions[0])[predicted_class].numpy()

        target_class = class_idx if class_idx is not None else predicted_class

        # Compute LRP
        attribution = self.compute_lrp(processed_image, class_idx=target_class)

        # Apply heat quantization
        attribution_quant = heat_quantization(attribution, num_bins=8)
        # Results
        results = {
            "predicted_class": predicted_class,
            "target_class": target_class,
            "confidence": confidence,
            "attribution_map": attribution,
            "attribution_map_quantized": attribution_quant,
            "display_image": display_image,
            "processed_image": processed_image,
        }

        # Visualization
        if show_plot:
            self._visualize_results(results)

        return results

    def _visualize_results(self, results):
        """Visualize analysis results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(results["display_image"])
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Attribution map
        im = axes[1].imshow(results["attribution_map_quantized"], cmap="RdBu_r")
        axes[1].set_title("LRP Attribution")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        # Overlay
        axes[2].imshow(results["display_image"])
        axes[2].imshow(results["attribution_map"], cmap="RdBu_r", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.suptitle(
            f"Class: {results['target_class']}, Confidence: {results['confidence']:.3f}"
        )
        plt.tight_layout()
        plt.show()


def load_sample_image(image_path: Optional[str] = None) -> tf.Tensor:
    """Load a sample image for testing"""
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        print(f"Loaded image from {image_path}")
        return image

    except Exception as e:
        print(f"Failed to load image: {e}")
        print("Creating synthetic test image")

        # Create synthetic image
        image = tf.zeros((224, 224, 3), dtype=tf.float32)
        center_x, center_y = 112, 112
        y_coords, x_coords = tf.meshgrid(
            tf.range(224, dtype=tf.float32),
            tf.range(224, dtype=tf.float32),
            indexing="ij",
        )

        dist = tf.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        circle_mask = dist < 50

        image = tf.stack(
            [
                tf.where(circle_mask, 255.0 * (1 - dist / 50), 50.0),
                tf.where(dist < 80, 255.0 * tf.sin(dist / 10), 100.0),
                tf.where(dist > 80, 255.0 * (dist - 80) / 80, 150.0),
            ],
            axis=-1,
        )

        image = tf.clip_by_value(image, 0.0, 255.0)
        return image


def heat_quantization(attribution_map: tf.Tensor, num_bins: int = 8) -> tf.Tensor:
    """
    Apply heat quantization to attribution map as described in the paper

    Formula: α_{i,j} = (α_R)_min + floor((α_R)_{i,j} - (α_R)_min) / ((α_R)_max - (α_R)_min) / Q) * Q

    Args:
        attribution_map: Raw attribution map
        num_bins: Number of quantization bins (Q in the paper)

    Returns:
        Quantized attribution map
    """
    min_val = tf.reduce_min(attribution_map)
    max_val = tf.reduce_max(attribution_map)

    # Calculate bin width
    range_val = max_val - min_val
    bin_width = range_val / tf.cast(num_bins, tf.float32)

    # Apply quantization formula from the paper
    normalized = (attribution_map - min_val) / bin_width
    binned = tf.floor(normalized)
    quantized = min_val + (binned * bin_width)

    return quantized


def demo_clean_lrp():
    """Demonstration of the clean LRP implementation"""
    print("=" * 50)
    print("Clean LRP ResNet50 Demo")
    print("=" * 50)

    # Create model
    print("Creating Clean LRP ResNet50...")
    model = CleanLRPResNet50(weights="imagenet")

    # Load sample image
    print("Loading sample image...")
    image = load_sample_image("n01855032_red-breasted_merganser.JPEG")

    # Analyze image
    print("Analyzing image with LRP...")
    results = model.analyze_image(image, show_plot=True)

    print(f"\nAnalysis Results:")
    print(f"Predicted class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Attribution map shape: {results['attribution_map'].shape}")
    print(
        f"Attribution range: [{tf.reduce_min(results['attribution_map']):.3f}, {tf.reduce_max(results['attribution_map']):.3f}]"
    )

    return model, results


if __name__ == "__main__":
    demo_clean_lrp()
