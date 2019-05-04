import keras
import os
import argparse

from keras.optimizers import SGD, Adam
from keras.layers import Activation, GlobalAveragePooling2D, Dense
from keras.models import Model

from data.openimages.batch_generator_db import OpenImagesData
from data.openimages.constants import Constants as OpenImagesDataConstants, DatasetTypes
from keras_custom.callbacks.model_saver import ModelSaver
from keras_custom.callbacks.training_progress_db_updater import TrainingProgressDbUpdater
from keras.callbacks import TensorBoard

from utils.get_last_checkpoint_from_dir import get_checkpoint_for_retraining
from utils.copy_file_to_scratch import copy_file_to_scratch
from utils.get_job_id import get_job_id
from checkpoints.utils import make_checkpoint_model_name
from tensorflow.python.client import device_lib
from keras_custom.metrics.f1_score import f1
from keras_custom.losses.masked_binary_crossentropy import make_masked_binary_cross_entropy
from common.argparse_types import str2bool
from datetime import datetime
from training_utils.db import TrainingUtilsDB
from training_utils.cli_helpers import load_training_session


class AvailableModelNames:
    inceptionV3 = 'inceptionV3'
    densenet121 = 'densenet121'


class AvailableOptimizers:
    sgd = 'sgd'
    adam = 'adam'


def load_inceptionV3_model(args: argparse.Namespace):
    training_session = load_training_session(args, DatasetTypes.MULTILABEL_CLASSIFICATION)

    training_utils_db = TrainingUtilsDB()
    training_run_config = training_utils_db.save_training_run_configuration(
        training_session_id=training_session.id,
        save_checkpoint_every_n_minutes=args.save_checkpoint_every_n_minutes,
        job_id= get_job_id(),
        optimizer_lr= args.optimizer_lr,
        generator_max_queue_size=args.generator_max_queue_size,
        continue_training_allowed_different_config_keys=args.continue_training_allowed_different_config_keys,
        continue_from_last_checkpoint=int(args.continue_from_last_checkpoint),
        last_checkpoint_path=None,
        tensorboard_monitor_freq=args.tensorboard_monitor_freq,
        copy_db_to_scratch=args.copy_db_to_scratch,
        use_multiprocessing=args.use_multiprocessing,
        validation_data_use_percentage=args.validation_data_use_percentage,
        workers=args.workers,
        use_gpu=int(args.use_gpu)
    )

    checkpoint_dir = training_session.model_checkpoints_path
    file_for_latest_model = get_checkpoint_for_retraining(checkpoint_dir)

    if file_for_latest_model and args.continue_from_last_checkpoint:
        print(f'Loading model from {file_for_latest_model}')
        print(f'Num of top layers unfreezing: {training_session.unfreeze_top_k_layers}')

        model_checkpoint_path = os.path.join(checkpoint_dir, file_for_latest_model)
        # retrained_model.load_weights(model_checkpoint_path)

        if training_session.use_multitarget_learning:
            retrained_model = keras.models.load_model(model_checkpoint_path, {
                'f1': f1,
                'masked_binary_cross_entropy': make_masked_binary_cross_entropy(
                    OpenImagesDataConstants.MASK_VALUE_MISSING_LABEL)
            })
        else:
            retrained_model = keras.models.load_model(model_checkpoint_path, {
                'f1': f1
            })

            print(f'Loaded model from {model_checkpoint_path}. Going to continue training.')

        return retrained_model, training_session, training_run_config

    model = keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None
    )

    num_of_top_k_trainable_layers = training_session.unfreeze_top_k_layers
    if num_of_top_k_trainable_layers == 'all':
        num_of_top_k_trainable_layers = len(model.layers)
    else:
        num_of_top_k_trainable_layers = int(num_of_top_k_trainable_layers)

    print(f'Num of top layers unfreezing: {num_of_top_k_trainable_layers}')
    for pretrained_layer in model.layers:
        pretrained_layer.trainable = False

    if num_of_top_k_trainable_layers > 0:
        for pretrained_layer in model.layers[-num_of_top_k_trainable_layers:]:
            pretrained_layer.trainable = True

    pooled = GlobalAveragePooling2D(
        name='global_avg_pooling_before_output')(model.layers[-1].output)
    dense = Dense(OpenImagesDataConstants.NUM_OF_TRAINABLE_CLASSES,
                  name='retrained_final_dense')(pooled)
    o_retrained = Activation('sigmoid', name='output_activation')(dense)
    retrained_model = Model(input=model.input, output=o_retrained)

    optimizer_kargs = {}
    if training_run_config.optimizer_lr:
        optimizer_kargs['lr'] = training_run_config.optimizer_lr

    optimizer = None
    if training_session.optimizer == AvailableOptimizers.sgd:
        optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    elif training_session.optimizer == AvailableOptimizers.adam:
        optimizer = Adam(**optimizer_kargs)

    for idx, layer in enumerate(retrained_model.layers):
        print(f'Layer with idx: {idx} is trainable: {layer.trainable}')

    loss = None
    if training_session.use_multitarget_learning:
        loss = make_masked_binary_cross_entropy(OpenImagesDataConstants.MASK_VALUE_MISSING_LABEL)
    else:
        loss = 'binary_crossentropy'

    retrained_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['top_k_categorical_accuracy', f1]
    )

    return retrained_model, training_session, training_run_config


def load_model(args):
    if args.model == AvailableModelNames.inceptionV3:
        return load_inceptionV3_model(args)
    else:
        raise Exception('Model not recognized in load_model().')


def perform_retraining(args):
    db_images_path = args.db_images_path
    db_image_labels_path = args.db_image_labels_path

    retrained_model, training_session, training_run_config = load_model(args)

    if training_run_config.copy_db_to_scratch:
        db_images_path = copy_file_to_scratch(db_images_path)
        db_image_labels_path = copy_file_to_scratch(db_image_labels_path)

    if not args.use_gpu:
        print("*** Using CPU")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print("*** Using GPU")

    BATCH_SIZE = training_session.batch_size
    TOTAL_NUM_OF_SAMPLES = training_session.num_of_training_images  # 1000000

    VALIDATION_NUM_OF_SAMPLES = training_session.num_of_validations_images * training_run_config.validation_data_use_percentage  # 41620

    NUM_OF_BATCHES_FOR_ONE_EPOCH = int(TOTAL_NUM_OF_SAMPLES / BATCH_SIZE)
    NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL = int(VALIDATION_NUM_OF_SAMPLES / BATCH_SIZE)

    datetime_training_start = datetime.utcnow()

    openimages_generator = OpenImagesData(
        batch_size=BATCH_SIZE,
        len=int(NUM_OF_BATCHES_FOR_ONE_EPOCH),
        num_of_classes=OpenImagesDataConstants.NUM_OF_TRAINABLE_CLASSES,
        db_images_path=db_images_path,
        db_image_labels_path=db_image_labels_path,
        total_number_of_samples=TOTAL_NUM_OF_SAMPLES,
        use_multitarget_learning=training_session.use_multitarget_learning
    )

    openimages_generator_val = OpenImagesData(
        batch_size=BATCH_SIZE,
        len=int(NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL),
        num_of_classes=OpenImagesDataConstants.NUM_OF_TRAINABLE_CLASSES,
        total_number_of_samples=VALIDATION_NUM_OF_SAMPLES,
        db_images_path=db_images_path,
        db_image_labels_path=db_image_labels_path,
        table_name_for_image_urls=OpenImagesDataConstants.VALIDATION_TABLE_NAME_IMAGES,
        use_multitarget_learning=training_session.use_multitarget_learning
    )

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
        training_session_id = training_session.id,
        training_run_config_id = training_run_config.id,
        metrics_to_save=[
            'f1',
            'top_k_categorical_accuracy',
            'val_f1',
            'val_top_k_categorical_accuracy',
            'loss',
            'val_loss'
        ],
        path_to_checkpoints=training_session.model_checkpoints_path
    )

    retrained_model.fit_generator(
        openimages_generator,
        epochs=args.epochs,
        callbacks=[checkpointer, tensorboard_cb, training_progress_updater_cb],
        use_multiprocessing=training_run_config.use_multiprocessing,
        workers=training_run_config.workers,
        max_queue_size=args.generator_max_queue_size,
        validation_data=openimages_generator_val,
        initial_epoch=training_session.num_of_epochs_processed
    )

    checkpoint_name = make_checkpoint_model_name(datetime_training_start, 'final')
    path_to_save_final_model = os.path.join(training_session.model_checkpoints_path, checkpoint_name)

    retrained_model.save(path_to_save_final_model)


def main():
    print(device_lib.list_local_devices())

    parser = argparse.ArgumentParser(description='Retrain pretrained vision network for openimages.')

    parser.add_argument('--model',
                        type=str,
                        choices=[AvailableModelNames.inceptionV3],
                        required=True,
                        help='Model to retrain.')

    parser.add_argument('--images_num',
                        type=int,
                        default=5000000,
                        help='Number of images to use for training.')

    parser.add_argument('--validation_images_num',
                        type=int,
                        default=40000,
                        help='Number of images to use for validation.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=30,
                        help='Batch size for model.')

    parser.add_argument('--validation_data_use_percentage',
                        type=float,
                        default=1.0,
                        help='Percentage of validation data to use.')

    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help='Number of workers to use.')

    parser.add_argument('--db_images_path',
                        type=str,
                        default=OpenImagesDataConstants.IMAGES_DB_PATH,
                        help='Path to database containing images.')

    parser.add_argument('--db_image_labels_path',
                        type=str,
                        default=OpenImagesDataConstants.IMAGE_LABELS_DB_PATH,
                        help='Path to database containing labels.')

    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='Num of epochs to train for.')

    parser.add_argument('--optimizer',
                        type=str,
                        choices=[AvailableOptimizers.sgd,
                                 AvailableOptimizers.adam],
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

    parser.add_argument('--training_session_id',
                        type=int,
                        default=None,
                        help='Training session id to continue from.')

    args = parser.parse_args()

    print(args)
    perform_retraining(args)


if __name__ == "__main__":
    main()
