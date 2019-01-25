import argparse

from keras.optimizers import SGD, Adam
from densenetpretrained import densenet121
from keras.layers import Activation, GlobalAveragePooling2D, Dense
from keras.models import Model
from densenetpretrained.custom_layers import Scale

from data.openimages.batch_generator_db import OpenImagesData
from data.openimages.constants import Constants as OpenImagesDataConstants
from callbacks.model_saver import ModelSaver

from utils.get_last_checkpoint_from_dir import get_checkpoint_for_retraining, get_last_checkpoint_from_dir
from utils.copy_file_to_scratch import copy_file_to_scratch
from utils.training_config_utils import save_training_config
from utils.get_job_id import get_job_id
from checkpoints.utils import make_checkpoint_model_dir, make_checkpoint_model_name
from tensorflow.python.client import device_lib
from keras_custom.metrics.f1_score import f1
from common.argparse_types import str2bool

import keras
import os
import re
from datetime import datetime

class AvailableModelNames:
  inceptionV3 = 'inceptionV3'
  densenet121 = 'densenet121'

class AvailableOptimizers:
  sgd = 'sgd'
  adam = 'adam'


def load_densenet121_model(args):
  checkpoint_dir = './checkpoints/%s/%s' % (args.model, str(args.images_num))
  file_for_latest_model = get_last_checkpoint_from_dir(checkpoint_dir)

  # Use pre-trained weights for Tensorflow backend
  weights_path = 'imagenet_models/densenet121_weights_tf.h5'

  # Test pretrained model
  model = densenet121.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

  sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',])

  model.layers.pop()
  model.layers.pop()
  model.layers.pop()

  pooled = GlobalAveragePooling2D(name='global_avg_pooling_before_output')(model.layers[-1].output)
  dense = Dense(OpenImagesDataConstants.NUM_OF_TRAINABLE_CLASSES, name='retrained_fc6')(pooled)
  o_retrained = Activation('sigmoid', name='output_activation')(dense)

  retrained_model = Model(input=model.input, output=[o_retrained])

  retrained_model.compile(
    optimizer=sgd,
    loss='binary_crossentropy',
    metrics=['top_k_categorical_accuracy', f1]
  )

  # retrained_model.summary()

  if file_for_latest_model:
    print("File for latest model %s" % file_for_latest_model)
    with keras.utils.CustomObjectScope({'Scale': Scale}):
      print("Loading model weight...")
      model_checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_dir, file_for_latest_model)
      retrained_model.load_weights(model_checkpoint_path)
      print("Model loaded...")


  return retrained_model


def load_inceptionV3_model(args: argparse.Namespace):
  checkpoint_dir = make_checkpoint_model_dir(args)
  file_for_latest_model = get_checkpoint_for_retraining(args)

  model = keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None
  )

  num_of_top_k_trainable_layers = args.unfreeze_top_k_layers

  print(f'Num of top layers unfreezing: {num_of_top_k_trainable_layers}')
  for pretrained_layer in model.layers:
    pretrained_layer.trainable = False

  if num_of_top_k_trainable_layers > 0:
    for pretrained_layer in model.layers[-num_of_top_k_trainable_layers:]:
      pretrained_layer.trainable = True

  pooled = GlobalAveragePooling2D(name='global_avg_pooling_before_output')(model.layers[-1].output)
  dense = Dense(OpenImagesDataConstants.NUM_OF_TRAINABLE_CLASSES, name='retrained_final_dense')(pooled)
  o_retrained = Activation('sigmoid', name='output_activation')(dense)
  retrained_model = Model(input=model.input, output=o_retrained)


  optimizer_kargs = {}
  if args.optimizer_lr:
    optimizer_kargs['lr'] = args.optimizer_lr

  optimizer = None
  if args.optimizer == AvailableOptimizers.sgd:
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  elif args.optimizer == AvailableOptimizers.adam:
    optimizer = Adam(**optimizer_kargs)


  for idx, layer in enumerate(retrained_model.layers):
    print(f'Layer with idx: {idx} is trainable: {layer.trainable}')

  if file_for_latest_model and args.continue_from_last_checkpoint:
    print(f'Loading model from {file_for_latest_model}')
    model_checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_dir, file_for_latest_model)
    # retrained_model.load_weights(model_checkpoint_path)

    retrained_model = keras.models.load_model(model_checkpoint_path, {
      'f1': f1,
    })
  else:
    retrained_model.compile(
      optimizer=optimizer,
      loss='binary_crossentropy',
      metrics=['top_k_categorical_accuracy', f1]
    )



  return retrained_model

def load_model(args):
  if args.model == AvailableModelNames.inceptionV3:
    return load_inceptionV3_model(args)
  elif args.model == AvailableModelNames.densenet121:
    return load_densenet121_model(args)
  else:
    raise Exception('Model not recognized in load_model().')


def perform_retraining(args):
  db_images_path = args.db_images_path
  db_image_labels_path = args.db_image_labels_path

  if args.copy_db_to_scratch:
    db_images_path = copy_file_to_scratch(db_images_path)
    db_image_labels_path = copy_file_to_scratch(db_image_labels_path)

  if args.use_gpu == False:
    print("*** Using CPU")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    print("*** Using GPU")

  retrained_model = load_model(args)

  checkpoint_dir = make_checkpoint_model_dir(args)

  BATCH_SIZE = args.batch_size
  TOTAL_NUM_OF_SAMPLES = args.images_num  # 1000000

  VALIDATION_DATA_USE_PERCENTAGE = args.validation_data_use_percentage
  VALIDATION_NUM_OF_SAMPLES = args.validation_images_num * VALIDATION_DATA_USE_PERCENTAGE  # 41620

  NUM_OF_BATCHES_FOR_ONE_EPOCH = int(TOTAL_NUM_OF_SAMPLES / BATCH_SIZE)
  NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL = int(VALIDATION_NUM_OF_SAMPLES / BATCH_SIZE)

  NUM_OF_WORKERS = args.workers

  datetime_training_start = datetime.utcnow()

  openimages_generator = OpenImagesData(
    batch_size=BATCH_SIZE,
    len=int(NUM_OF_BATCHES_FOR_ONE_EPOCH),
    num_of_classes=OpenImagesDataConstants.NUM_OF_CLASSES,
    db_images_path=db_images_path,
    db_image_labels_path=db_image_labels_path,
    total_number_of_samples=TOTAL_NUM_OF_SAMPLES
  )

  openimages_generator_val = OpenImagesData(
    batch_size=BATCH_SIZE,
    len=int(NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL),
    num_of_classes=OpenImagesDataConstants.NUM_OF_CLASSES,
    total_number_of_samples=VALIDATION_NUM_OF_SAMPLES,
    db_images_path=db_images_path,
    db_image_labels_path=db_image_labels_path,
    table_name_for_image_urls='validation_images'
  )

  checkpointer = ModelSaver(
    every_n_minutes=args.save_checkpoint_every_n_minutes,
    checkpoint_model_dir=checkpoint_dir,
    train_args=args,
    datetime_start=datetime_training_start
  )


  save_training_config(args, checkpoint_dir, get_job_id())

  retrained_model.fit_generator(
    openimages_generator,
    epochs=args.epochs,
    callbacks=[checkpointer],
    use_multiprocessing=True,
    workers=NUM_OF_WORKERS,
    # validation_data=openimages_generator_val
  )

  checkpoint_name = make_checkpoint_model_name(datetime_training_start, 'final')
  path_to_save_final_model = os.path.join(checkpoint_dir, checkpoint_name)

  retrained_model.save(path_to_save_final_model)

def main():
  print(device_lib.list_local_devices())

  parser = argparse.ArgumentParser(description='Retrain pretrained vision network for openimages.')

  parser.add_argument('--model',
                      type=str,
                      choices=[AvailableModelNames.inceptionV3, AvailableModelNames.densenet121],
                      required=True,
                      help='Model to retrain.')

  parser.add_argument('--images_num',            
                      type=int,
                      default = 5000000,
                      help='Number of images to use for training.')

  parser.add_argument('--validation_images_num', 
                      type=int,
                      default = 40000,
                      help='Number of images to use for validation.')

  parser.add_argument('--batch_size', 
                      type=int,
                      default=30,
                      help='Batch size for model.')

  parser.add_argument('--validation_data_use_percentage', 
                      type=float, 
                      default = 0.0001,
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
                      choices=[AvailableOptimizers.sgd, AvailableOptimizers.adam],
                      required=True, 
                      help='Optimizer to use.')

  parser.add_argument('--optimizer_lr', 
                      type=float, 
                      required=False, 
                      help='Learning rate for optimizer to use.')

  parser.add_argument('--unfreeze_top_k_layers',
                      type=int,
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


  args = parser.parse_args()

  print(args)
  perform_retraining(args)


if __name__ == "__main__":
  main()
