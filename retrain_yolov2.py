import tensorflow as tf
import numpy as np
import asyncio
import argparse

from data.openimages.constants import BoxableImagesConstants
from models.yolov2.constants import Constants as YOLOV2Constants
from keras.models import load_model, Model
from keras.layers import Conv2D, Lambda, Input
from models.yolov2.loss import lambda_loss
from models.yolov2.utils.load_anchors import load_anchors
from models.yolov2.preprocessing.training import preprocess_image_bytes, preprocess_true_boxes
from data.openimages.boxable_db import async_get_boxes_by_image_ids, async_get_images_by_ids
from models.yolov2.preprocessing.training import get_detector_mask
from data.openimages.boxable_batch_generator_db import BoxableOpenImagesData
from common.argparse_types import str2bool
from utils.copy_file_to_scratch import copy_file_to_scratch


class AvailableOptimizers:
    sgd = 'sgd'
    adam = 'adam'


def get_model_for_retraining():
    anchors = load_anchors()
    num_classes = BoxableImagesConstants.NUM_OF_CLASSES
    path_to_pretrained_yolov2 = YOLOV2Constants.PATH_TO_PRETRAINED_YOLO_V2_MODEL_FULLY_CONV

    pretrained_model = load_model(path_to_pretrained_yolov2)

    # topless_pretrained_yolo_v2.layers.pop()

    yolov2_top = Conv2D(
        filters = len(anchors) * (5 + num_classes),
        kernel_size=(1, 1),
        activation='linear',
        name='final_conv2d'
    )(pretrained_model.layers[-2].output)

    yolov2_body = Model(pretrained_model.layers[0].input, yolov2_top)

    # Create model input layers.
    boxes_input = Input(shape=(None, 5), name='boxes_input')
    detectors_mask_shape = (14, 14, 5, 1)
    matching_boxes_shape = (14, 14, 5, 5)

    detectors_mask_input = Input(shape=detectors_mask_shape, name="detectors_mask_input")
    matching_boxes_input = Input(shape=matching_boxes_shape, name="matching_boxes_input")

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        yolo_loss_layer = Lambda(
            lambda_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={
                'anchors': anchors,
                'num_classes': num_classes
            })

        model_loss = yolo_loss_layer([
            yolov2_body.output, 
            boxes_input,
            detectors_mask_input,
            matching_boxes_input
        ])


    yolov2_model_with_loss = Model([yolov2_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)

    yolov2_model_with_loss.compile(optimizer='adam')

    return yolov2_body, yolov2_model_with_loss


async def main(args):
    yolov2_body, yolov2_model_with_loss = get_model_for_retraining()

    num_of_top_k_trainable_layers = args.unfreeze_top_k_layers

    db_path = args.db_path
    if args.copy_db_to_scratch:
        db_path = copy_file_to_scratch(db_path)

    print(f'Num of top layers unfreezing: {num_of_top_k_trainable_layers}')
    for pretrained_layer in yolov2_model_with_loss.layers:
        pretrained_layer.trainable = False

    if args.unfreeze_top_k_layers == 'all':
        for pretrained_layer in yolov2_model_with_loss.layers:
            pretrained_layer.trainable = True
    else:
        # last two layers must be always trainable
        for pretrained_layer in yolov2_model_with_loss.layers[-(num_of_top_k_trainable_layers + 2):]:
            pretrained_layer.trainable = True

    # This is a hack to use the custom loss function in the last layer
    yolov2_model_with_loss.compile( optimizer='adam', loss={ 'yolo_loss': lambda y_true, y_pred: y_pred }) 

    for idx, pretrained_layer in enumerate(yolov2_model_with_loss.layers):
        print(f'Layer with idx is trainable: {pretrained_layer.trainable}')

    # logging = TensorBoard()
    # checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
    #                              save_weights_only=True, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # yolov2_model_with_loss.fit([preprocessed_image_bytes, preprocessed_image_boxes, detectors_mask, matching_true_boxes],
    #     np.zeros(len(preprocessed_image_bytes)),
    #     batch_size=32,
    #     epochs=5)

    generator = BoxableOpenImagesData(
        db_path=db_path,
        table_name_for_images=args.table_name_images,
        table_name_for_image_boxes=args.table_name_image_boxes,
        batch_size=args.batch_size,
        len=args.images_num // args.batch_size,
        num_of_classes=BoxableImagesConstants.NUM_OF_CLASSES,
        total_number_of_samples=args.images_num,
        use_multitarget_learning=args.use_multitarget_learning,
    )

    yolov2_model_with_loss.fit_generator(
        generator,
        epochs=args.epochs,
        use_multiprocessing=args.use_multiprocessing,
        workers=args.workers,
        max_queue_size=args.generator_max_queue_size,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for retraining YOLOv2 on OpenImages boxable dataset.")

    parser.add_argument('--db_path',
                        type=str,
                        default=BoxableImagesConstants.PATH_TO_DB_YOLO_V2,
                        help='Path to database containing boxable images from OpenImages.')

    parser.add_argument('--table_name_images',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_TRAIN_BOXABLE_IMAGES,
                        help='Table name for images.')

    parser.add_argument('--table_name_image_boxes',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_TRAIN_IMAGE_BOXES,
                        help='Table name for image boxes.')

    parser.add_argument('--images_num',
                        type=int,
                        default=BoxableImagesConstants.NUM_OF_TRAIN_SAMPLES,
                        help='Number of images to use for training.')

    parser.add_argument('--validation_images_num',
                        type=int,
                        default=40000,
                        help='Number of images to use for validation.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size for model.')

    parser.add_argument('--validation_data_use_percentage',
                        type=float,
                        default=1.0,
                        help='Percentage of validation data to use.')

    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help='Number of workers to use.')

    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='Num of epochs to train for.')

    parser.add_argument('--optimizer',
                        type=str,
                        choices=[
                            AvailableOptimizers.sgd,
                            AvailableOptimizers.adam
                        ],
                        required=True,
                        help='Optimizer to use.')

    parser.add_argument('--optimizer_lr',
                        type=float,
                        required=False,
                        help='Learning rate for optimizer to use.')

    parser.add_argument('--unfreeze_top_k_layers',
                        default=0,
                        required=False,
                        help='Number of top k layers from pretrained model to unfreeze.')

    parser.add_argument('--copy_db_to_scratch',
                        type=str2bool,
                        default=True,
                        help='Flag to determine whether copy db files to scratch.')

    parser.add_argument('--save_checkpoint_every_n_minutes',
                        type=int,
                        default=118,
                        help='Specify how often to save trained model.')

    parser.add_argument('--use_gpu',
                        type=str2bool,
                        default=True,
                        help='Specify whether use GPU')

    parser.add_argument('--continue_training_allowed_different_config_keys',
                        type=str,
                        default='epochs,db_images_path,db_image_labels_path',
                        help='Comma-separated list of values which are allowed to be different for checkpoint to continue training.')

    parser.add_argument('--tensorboard_monitor_freq',
                        type=int,
                        default=0,
                        help='Number specifying after how many processed examples to log the metrics for tensorboard. 0 means disable logging...')

    parser.add_argument('--continue_from_last_checkpoint',
                        type=str2bool,
                        default=True,
                        help='Specify whether you want to continue from last checkpoint.')

    parser.add_argument('--use_multitarget_learning',
                        type=str2bool,
                        default=True,
                        help='Specify whether you want to use multitarget learning objective(masked binary crossentropy).')

    parser.add_argument('--generator_max_queue_size',
                        type=int,
                        default=100,
                        help='Max queue size for generator.')

    parser.add_argument('--use_multiprocessing',
                        type=str2bool,
                        default=True,
                        help='Specify whether you want to use multiple processes for loading data (otherwise threads will be used).')

    args = parser.parse_args()

    event_loop = asyncio.get_event_loop()

    event_loop.run_until_complete(main(args))
