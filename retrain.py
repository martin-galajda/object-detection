import argparse

from keras.optimizers import SGD
from densenetpretrained import densenet121
from keras.layers import Activation, GlobalAveragePooling2D, Dense
from keras.models import Model
from densenetpretrained.custom_layers import Scale

from data.openimages.batch_generator_db import OpenImagesData
from data.openimages.constants import Constants as OpenImagesDataConstants
from callbacks.model_saver import ModelSaver

from utils.get_last_checkpoint_from_dir import get_last_checkpoint_from_dir

import keras
import os


class AvailableModelNames:
  inceptionV3 = 'inceptionV3'
  densenet121 = 'densenet121'


def load_densenet121_model(args):
  checkpoint_dir = './checkpoints/%s/%s' % (args.model, str(args.images_num))
  file_for_latest_model = get_last_checkpoint_from_dir(checkpoint_dir)

  # Use pre-trained weights for Tensorflow backend
  weights_path = 'imagenet_models/densenet121_weights_tf.h5'

  # Test pretrained model
  model = densenet121.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

  sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  model.layers.pop()
  model.layers.pop()
  model.layers.pop()

  pooled = GlobalAveragePooling2D(name='global_avg_pooling_before_output')(model.layers[-1].output)
  dense = Dense(OpenImagesDataConstants.NUM_OF_CLASSES, name='retrained_fc6')(pooled)
  o_retrained = Activation('sigmoid', name='output_activation')(dense)

  retrained_model = Model(input=model.input, output=[o_retrained])

  retrained_model.compile(optimizer=sgd, loss='binary_crossentropy',
                          metrics=['binary_accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'])

  # retrained_model.summary()

  if file_for_latest_model:
    print("File for latest model %s" % file_for_latest_model)
    with keras.utils.CustomObjectScope({'Scale': Scale}):
      print("Loading model weight...")
      model_checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_dir, file_for_latest_model)
      retrained_model.load_weights(model_checkpoint_path)
      print("Model loaded...")


  return retrained_model


def load_inceptionV3_model(args):
  checkpoint_dir = './checkpoints/%s/%s' % (args.model, str(args.images_num))
  file_for_latest_model = get_last_checkpoint_from_dir(checkpoint_dir)

  model = keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None
  )

  model.summary()
  pooled = GlobalAveragePooling2D(name='global_avg_pooling_before_output')(model.layers[-1].output)
  dense = Dense(OpenImagesDataConstants.NUM_OF_CLASSES, name='retrained_final_dense')(pooled)
  o_retrained = Activation('sigmoid', name='output_activation')(dense)
  retrained_model = Model(input=model.input, output=o_retrained)

  sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
  retrained_model.compile(
    optimizer=sgd,
    loss='binary_crossentropy',
    metrics=['binary_accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy']
  )

  if file_for_latest_model:
    print("File for latest model %s" % file_for_latest_model)
    model_checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_dir, file_for_latest_model)
    retrained_model.load_weights(model_checkpoint_path)

  return retrained_model

def load_model(args):
  if args.model == AvailableModelNames.inceptionV3:
    return load_inceptionV3_model(args)
  elif args.model == AvailableModelNames.densenet:
    return load_densenet121_model(args)
  else:
    raise Exception('Model not recognized in load_model().')



def perform_retraining(args):
  retrained_model = load_model(args)

  checkpoint_dir = './checkpoints/%s/%s' % (args.model, str(args.images_num))

  BATCH_SIZE = args.batch_size
  TOTAL_NUM_OF_SAMPLES = args.images_num  # 1000000

  VALIDATION_DATA_USE_PERCENTAGE = args.validation_data_use_percentage
  VALIDATION_NUM_OF_SAMPLES = args.validation_images_num * VALIDATION_DATA_USE_PERCENTAGE  # 41620

  NUM_OF_BATCHES_FOR_ONE_EPOCH = int(TOTAL_NUM_OF_SAMPLES / BATCH_SIZE)
  NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL = int(VALIDATION_NUM_OF_SAMPLES / BATCH_SIZE)

  NUM_OF_WORKERS = args.workers

  openimages_generator = OpenImagesData(
    batch_size=BATCH_SIZE,
    len=int(NUM_OF_BATCHES_FOR_ONE_EPOCH),
    num_of_classes=OpenImagesDataConstants.NUM_OF_CLASSES,
    db_images_path=args.db_images_path,
    db_image_labels_path=args.db_image_labels_path,
    total_number_of_samples=TOTAL_NUM_OF_SAMPLES
  )

  openimages_generator_val = OpenImagesData(
    batch_size=BATCH_SIZE,
    len=int(NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL),
    num_of_classes=OpenImagesDataConstants.NUM_OF_CLASSES,
    total_number_of_samples=VALIDATION_NUM_OF_SAMPLES,
    db_images_path=args.db_images_path,
    db_image_labels_path=args.db_image_labels_path,
    table_name_for_image_urls='validation_images'
  )

  checkpointer = ModelSaver(every_n_minutes=118, model=checkpoint_dir)


  retrained_model.fit_generator(
    openimages_generator,
    epochs=30,
    callbacks=[checkpointer],
    use_multiprocessing=True,
    workers=NUM_OF_WORKERS,
    # validation_data=openimages_generator_val
  )

def main():
  from tensorflow.python.client import device_lib
  print(device_lib.list_local_devices())

  parser = argparse.ArgumentParser(description='Retrain pretrained vision network for openimages.')

  parser.add_argument('--model', type=str, choices=[AvailableModelNames.inceptionV3, AvailableModelNames.densenet121],
                      help='Model do retrain.')
  parser.add_argument('--images_num', type=int, default= 5000000,
                      help='Number of images to use for training.')

  parser.add_argument('--validation_images_num', type=int, default= 40000,
                      help='Number of images to use for validation.')

  parser.add_argument('--batch_size', type=int, default=30,
                      help='Batch size for model.')

  parser.add_argument('--validation_data_use_percentage', type=float, default=0.0001,
                      help='Percentage of validation data to use.')

  parser.add_argument('--workers', type=int, default=10,
                      help='Number of workers to use.')

  parser.add_argument('--db_images_path', type=str, default=OpenImagesDataConstants.IMAGES_DB_PATH,
                      help='Path to database containing images.')

  parser.add_argument('--db_image_labels_path', type=str, default=OpenImagesDataConstants.IMAGE_LABELS_DB_PATH,
                      help='Path to database containing labels.')

  args = parser.parse_args()

  print(args)
  perform_retraining(args)


if __name__ == "__main__":
  main()
