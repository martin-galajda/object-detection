import asyncio
import argparse
import keras

from data.openimages.constants import BoxableImagesConstants, DatasetTypes
from data.openimages.boxable_batch_generator_db import BoxableOpenImagesData
from common.argparse_types import str2bool
from models.yolov2.loss import yolo_loss_fn
from utils.copy_file_to_scratch import copy_file_to_scratch

from models.yolov2.utils.get_model_for_retraining import get_model_for_retraining
from training_utils.cli_helpers import load_training_session, create_training_run
from keras.callbacks import TensorBoard
from keras_custom.callbacks.model_saver import ModelSaver
from keras_custom.callbacks.training_progress_db_updater import TrainingProgressDbUpdater
from training_utils.db_tables_field_names import TrainingSessionFields, TrainingRunConfigFields
from utils.get_last_checkpoint_from_dir import get_checkpoint_for_retraining
from datetime import datetime


class AvailableOptimizers:
    sgd = 'sgd'
    adam = 'adam'



def load_model(training_session: TrainingSessionFields, training_run_config: TrainingRunConfigFields):
    file_for_latest_model = None
    if training_run_config.continue_from_last_checkpoint:
        checkpoint_dir = training_session.model_checkpoints_path
        file_for_latest_model = get_checkpoint_for_retraining(checkpoint_dir)

    if file_for_latest_model:
        restored_model = keras.models.load_model(file_for_latest_model, {
            'yolo_loss': lambda y_true, y_pred: y_pred[0]
        })

        print(f'Loaded model from {model_checkpoint_path}. Going to continue training.')

        return restored_model

    # yolov2_body, yolov2_model_with_loss = get_model_for_retraining()
    yolov2_body = get_model_for_retraining()

    if training_session.unfreeze_top_k_layers == 'all':
        # num_of_top_k_trainable_layers = len(yolov2_model_with_loss.layers)
        num_of_top_k_trainable_layers = len(yolov2_body.layers)
    else:
        num_of_top_k_trainable_layers = int(training_session.unfreeze_top_k_layers)

    # for pretrained_layer in yolov2_model_with_loss.layers:
    for pretrained_layer in yolov2_body.layers:
        pretrained_layer.trainable = False

    # for pretrained_layer in yolov2_model_with_loss.layers[-num_of_top_k_trainable_layers:]:
    for pretrained_layer in yolov2_body.layers[-num_of_top_k_trainable_layers:]:
        pretrained_layer.trainable = True

    def yolo_lambda_loss_fn(y_true, y_pred):
        return y_pred

    # This is a hack to use the custom loss function in the last layer
    # yolov2_model_with_loss.compile(optimizer='adam', loss={
    #     'yolo_loss': yolo_lambda_loss_fn
    # })
    yolov2_body.compile(optimizer='adam', loss={
        'yolo_output': yolo_loss_fn
    })

    # for idx, pretrained_layer in enumerate(yolov2_model_with_loss.layers):
    for idx, pretrained_layer in enumerate(yolov2_body.layers):
        print(f'Layer with idx {idx} is trainable: {pretrained_layer.trainable}')

    # return yolov2_model_with_loss
    return yolov2_body


async def main(args):
    training_session = load_training_session(args, DatasetTypes.OBJECT_DETECTION)
    training_run_config = create_training_run(args, training_session)
    yolov2_model_with_loss = load_model(training_session, training_run_config)

    db_path = args.db_path
    if args.copy_db_to_scratch:
        db_path = copy_file_to_scratch(db_path)

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

    datetime_training_start = datetime.utcnow()
    checkpointer = ModelSaver(
        every_n_minutes=training_run_config.save_checkpoint_every_n_minutes,
        checkpoint_model_dir=training_session.model_checkpoints_path,
        training_run_config=training_run_config,
        datetime_start=datetime_training_start
    )

    tensorboard_cb = TensorBoard(
        log_dir=training_session.tensorboard_logs_path
    )
    training_progress_updater_cb = TrainingProgressDbUpdater(
        start=datetime_training_start,
        training_session_id=training_session.id,
        training_run_config_id=training_run_config.id,
        metrics_to_save=[
            'loss',
            'val_loss',
            'basic_accuracy',
            'val_basic_accuracy'
        ],
        path_to_checkpoints=training_session.model_checkpoints_path
    )

    yolov2_model_with_loss.fit_generator(
        generator,
        epochs=args.epochs,
        use_multiprocessing=args.use_multiprocessing,
        workers=args.workers,
        max_queue_size=args.generator_max_queue_size,
        callbacks=[checkpointer, tensorboard_cb, training_progress_updater_cb],
        # metrics=[]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for retraining YOLOv2 on OpenImages boxable dataset.")

    parser.add_argument('--db_path',
                        type=str,
                        default=BoxableImagesConstants.PATH_TO_DB_YOLO_V2,
                        help='Path to database containing boxable images from OpenImages.')

    parser.add_argument('--training_session_id',
                        type=int,
                        default=None,
                        help='Training session id to resume.')

    parser.add_argument('--model',
                        type=str,
                        default=BoxableImagesConstants.YOLOV2_MODEL_NAME,
                        choices=[
                            BoxableImagesConstants.YOLOV2_MODEL_NAME,
                        ],
                        help='Name of the model to use.')

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
