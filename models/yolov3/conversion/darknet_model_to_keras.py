import numpy as np
import os
from keras.layers import (
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    Add
)
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model as plot
from models.yolov3.conversion.utils import parse_darknet_config
from models.yolov3.conversion.constants import YoloV3Sections
from models.yolov3.conversion.reconstruct_conv_layer import parse_conv_layer


def parse_weights_file(weights_path: str):
    weights_file = open(weights_path, 'rb')

    major = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    minor = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    revision = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))

    print(f'Weights Header: major, minor, revision, seen images={major}, {minor}, {revision}, {seen}.')

    return weights_file

def convert_model(
    config_path: str,
    weights_path: str,
    output_path: str,
    *,
    plot_model: bool = False,
    path_to_graph_output: str = None
):
    output_root = os.path.splitext(output_path)[0]

    # Load weights and Darknet config.
    print('Loading weights from serialized binary file.')
    weights_file = parse_weights_file(weights_path)

    print('Parsing Darknet config.')
    cfg_parser = parse_darknet_config(config_path)

    print('Creating Keras model.')
    prev_layer = Input(shape=(None, None, 3))
    all_layers = [prev_layer]
    yolo_heads = []
    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4

    weights_read_total = 0
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith(YoloV3Sections.CONVOLUTIONAL):
            parsed_layer, weights_read_to_conv_layer = parse_conv_layer(
                prev_layer=prev_layer,
                layer_config=cfg_parser[section],
                weights_file=weights_file,
                weight_decay=weight_decay
            )
            all_layers.append(parsed_layer)
            prev_layer = parsed_layer
            weights_read_total += weights_read_to_conv_layer

        elif section.startswith(YoloV3Sections.MAX_POOL):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])

            parsed_layer = MaxPooling2D(
                padding='same',
                pool_size=(size, size),
                strides=(stride, stride)
            )(prev_layer)
            all_layers.append(parsed_layer)
            prev_layer = parsed_layer

        elif section.startswith(YoloV3Sections.AVG_POOL):
            parsed_layer = GlobalAveragePooling2D()(prev_layer)
            all_layers.append(parsed_layer)
            prev_layer = parsed_layer

        elif section.startswith(YoloV3Sections.ROUTE):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]

            if len(layers) > 1:
                concatenate_layer = concatenate(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                # only one layer to route
                skip_layer = layers[0]
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith(YoloV3Sections.UPSAMPLE):
            stride = cfg_parser[section]['stride']
            parsed_layer = UpSampling2D(size=(stride, stride), interpolation='bilinear')(prev_layer)
            all_layers.append(
                parsed_layer
            )
            prev_layer = parsed_layer

        elif section.startswith(YoloV3Sections.SHORTCUT):
            from_idx = cfg_parser[section]['from']
            from_layer = all_layers[int(from_idx)]
            parsed_layer = Add()([from_layer, prev_layer])
            all_layers.append(
                parsed_layer
            )
            prev_layer = parsed_layer

        elif section.startswith(YoloV3Sections.YOLO):
            yolo_layer = Lambda(lambda x: x, name=f'yolo_{len(yolo_heads)}')(prev_layer)
            all_layers.append(yolo_layer)
            yolo_heads += [yolo_layer]
            prev_layer = all_layers[-1]

        elif (
            section.startswith(YoloV3Sections.NET)
            or section.startswith(YoloV3Sections.COST)
            or section.startswith(YoloV3Sections.SOFTMAX)
        ):
            continue  # Configs not currently handled during model definition.

        else:
            raise ValueError(f'Unsupported section header type: {section}')

    # Create and save model.
    model = Model(inputs=all_layers[0], outputs=yolo_heads)
    print(model.summary())

    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print(f'Warning: {remaining_weights} unused weights')

    model.save(f'{output_path}')
    print(f'Saved Keras model to {output_path}')
    # Check to see if all weights have been read.
    print(f'Read {weights_read_total} of {weights_read_total + remaining_weights} from Darknet weights.')

    if plot_model:
        if path_to_graph_output is None:
            path_to_graph_output = output_root
        plot(model, to_file=f'{path_to_graph_output}.png', show_shapes=True)
        print(f'Saved model plot to {path_to_graph_output}.png')
