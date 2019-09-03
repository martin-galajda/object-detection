import numpy as np
import os
from keras import backend as K
from keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    Add
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
from models.yolov3.conversion.utils import parse_darknet_config
from models.yolov3.conversion.constants import YoloV3Sections, YoloV3Activations


def convert_model(
    config_path: str,
    weights_path: str,
    output_path: str,
    *,
    fully_convolutional: bool,
    plot_model: bool = False,
    path_to_graph_output: str = None
):
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')

    major = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    minor = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    revision = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))

    print(f'Weights Header: major, minor, revision, seen images={major}, {minor}, {revision}, {seen}.')
    print('Parsing Darknet config.')

    cfg_parser = parse_darknet_config(config_path)

    print('Creating Keras model.')
    if fully_convolutional:
        print('Fully convolutional.')
        image_height, image_width = None, None
    else:
        image_height = int(cfg_parser['net_0']['height'])
        image_width = int(cfg_parser['net_0']['width'])
    prev_layer = Input(shape=(image_height, image_width, 3))
    all_layers = [prev_layer]
    yolo_heads = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith(YoloV3Sections.CONVOLUTIONAL):
            #
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            # padding='same' is equivalent to Darknet pad=1
            # padding = 'same' if pad == 1 else 'valid'
            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            # TODO: This assumes channel last dim_ordering.
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters,),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                # TODO: Keras BatchNormalization mistakenly refers to var
                # as std.
                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation == YoloV3Activations.LEAKY:
                pass  # Add advanced activation later.
            elif activation != YoloV3Activations.LINEAR:
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            if stride > 1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(( (1,0),(1,0) ))(prev_layer)
            # Create Conv2D layer
            conv_layer = (Conv2D(
                filters,
                (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == YoloV3Activations.LINEAR:
                all_layers.append(prev_layer)
            elif activation == YoloV3Activations.LEAKY:
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith(YoloV3Sections.MAX_POOL):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    padding='same',
                    pool_size=(size, size),
                    strides=(stride, stride))(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith(YoloV3Sections.AVG_POOL):
            if cfg_parser.items(section) != []:
                raise ValueError('{} with params unsupported.'.format(section))
            all_layers.append(GlobalAveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith(YoloV3Sections.ROUTE):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]

            if len(layers) > 1:
                concatenate_layer = concatenate(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith(YoloV3Sections.UPSAMPLE):
            stride = cfg_parser[section]['stride']
            prev_layer = all_layers[-1]

            all_layers.append(
                UpSampling2D(size=(stride, stride), interpolation='bilinear')(prev_layer)
            )
            prev_layer = all_layers[-1]

        elif section.startswith(YoloV3Sections.SHORTCUT):
            from_idx = cfg_parser[section]['from']

            from_layer = all_layers[int(from_idx)]
            all_layers.append(
                Add()([from_layer, prev_layer])
            )
            prev_layer = all_layers[-1]



        elif section.startswith(YoloV3Sections.YOLO):
            prev_layer = all_layers[-1]
            yolo_layer = Lambda(lambda x: x, name=f'yolo_{len(yolo_heads)}')(prev_layer)
            all_layers.append(yolo_layer)
            yolo_heads += [yolo_layer]
            anchors = np.array(list(map(lambda x: int(x.strip()), cfg_parser[section]['anchors'].split(',')))).reshape(
                (9, 2))
            print(anchors)
            prev_layer = all_layers[-1]


        elif (
            section.startswith(YoloV3Sections.NET)
            or section.startswith(YoloV3Sections.COST)
            or section.startswith(YoloV3Sections.SOFTMAX)
        ):
            continue  # Configs not currently handled during model definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    model = Model(inputs=all_layers[0], outputs=yolo_heads)
    print(model.summary())

    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print(f'Warning: {remaining_weights} unused weights')

    model.save(f'{output_path}')
    print(f'Saved Keras model to {output_path}')
    # Check to see if all weights have been read.
    print(f'Read {count} of {count + remaining_weights} from Darknet weights.')

    if plot_model:
        if path_to_graph_output is None:
            path_to_graph_output = output_root
        plot(model, to_file=f'{path_to_graph_output}.png', show_shapes=True)
        print(f'Saved model plot to {path_to_graph_output}.png')