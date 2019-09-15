from typing import Dict, Any
from keras import backend as K
import keras
import numpy as np
from keras.layers import (
    Conv2D,
    ZeroPadding2D,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from models.yolov3_gpu_head_v2.conversion.constants import YoloV3Activations


def parse_conv_layer(
    *,
    prev_layer: keras.layers.Layer,
    layer_config: Dict[str, Any],
    weight_decay: float,
    weights_file: Any
):
    weight_read = 0

    filters = int(layer_config['filters'])
    size = int(layer_config['size'])
    stride = int(layer_config['stride'])
    pad = int(layer_config['pad'])
    activation = layer_config['activation']
    batch_normalize = 'batch_normalize' in layer_config

    # padding = 'same' is equivalent to pad = 1 in Darknet
    padding = 'same' if pad == 1 and stride == 1 else 'valid'

    # Darknet serializes convolutional weights as:
    # [bias/beta, [gamma, mean, variance], conv_weights]
    prev_layer_shape = K.int_shape(prev_layer)

    # This assumes channel last dimension ordering.
    weights_shape = (size, size, prev_layer_shape[-1], filters)
    darknet_w_shape = (filters, weights_shape[2], size, size)
    weights_size = np.product(weights_shape)

    print('with batch normalization' if batch_normalize else '  ', activation, weights_shape)

    conv_bias = np.ndarray(
        shape=(filters,),
        dtype='float32',
        buffer=weights_file.read(filters * 4))

    weight_read += filters

    if batch_normalize:
        bn_weights = np.ndarray(
            shape=(3, filters),
            dtype='float32',
            buffer=weights_file.read(filters * 12))

        weight_read += 3 * filters

        bn_weight_list = [
            bn_weights[0],  # <-- scale gamma
            conv_bias,      # <-- shift beta
            bn_weights[1],  # <-- running mean
            bn_weights[2]   # <-- running var
        ]

    conv_weights = np.ndarray(
        shape=darknet_w_shape,
        dtype='float32',
        buffer=weights_file.read(weights_size * 4))
    weight_read += weights_size

    # DarkNet conv_weights are serialized Caffe-style:  (output_dim, input_dim, height, width)
    # We need to set them into Tensorflow order: (height, width, input_dim, output_dim)
    height_dim_pos, width_dim_pos, input_dim_pos, output_dim_pos = 2, 3, 1, 0
    conv_weights = np.transpose(conv_weights, [height_dim_pos, width_dim_pos, input_dim_pos, output_dim_pos])

    conv_weights = [conv_weights] if batch_normalize else [
        conv_weights,
        conv_bias
    ]

    if stride > 1:
        # Darknet uses left and top padding instead of 'same' mode
        prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)

    # Create Conv2D layer
    conv_layer = (Conv2D(
        filters,
        (size, size),
        strides=(stride, stride),
        kernel_regularizer=l2(weight_decay),
        use_bias=not batch_normalize,
        weights=conv_weights,
        activation=None,
        padding=padding))(prev_layer)

    if batch_normalize:
        conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)

    if activation == YoloV3Activations.LINEAR:
        return conv_layer, weight_read

    if activation == YoloV3Activations.LEAKY:
        conv_layer = LeakyReLU(alpha=0.1)(conv_layer)
        return conv_layer, weight_read

    raise ValueError(f'Unknown activation function `{activation}`.')
