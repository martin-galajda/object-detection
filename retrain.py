"""Test ImageNet pretrained DenseNet"""

import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K
from download_train_images import download_images
from keras.layers import MaxPooling2D
# We only test DenseNet-121 in this script for demo purpose
from densenetpretrained import densenet121
from keras.layers import Activation, GlobalAveragePooling2D, Dense
from keras.models import Model
from densenetpretrained.custom_layers import Scale
# from keras.callbacks import ModelCheckpoint

from data.openimages.batch_generator import OpenImagesData
from data.openimages.constants import Constants as OpenImagesDataConstants
from callbacks.model_saver import ModelSaver

import keras
import os
import re

files_in_models = os.listdir('./checkpoints')
files_in_models = [file for file in files_in_models if re.match('.*hdf5.*', file)]

file_for_latest_model = sorted(files_in_models)[-1]

print("File for latest model %s" % file_for_latest_model)
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

retrained_model.compile(optimizer=sgd, loss='binary_crossentropy')

# retrained_model.summary()


if file_for_latest_model:
  with keras.utils.CustomObjectScope({'Scale': Scale}):
    print("Loading model weight...")
    model_checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', file_for_latest_model)
    retrained_model.load_weights(model_checkpoint_path)
    print("Model loaded...")

BATCH_SIZE = 30
TOTAL_NUM_OF_SAMPLES = 5000000 # 1000000
VALIDATION_NUM_OF_SAMPLES = 40000 # 41620

NUM_OF_BATCHES_FOR_ONE_EPOCH = int(TOTAL_NUM_OF_SAMPLES / BATCH_SIZE)
NUM_OF_BATCHES_FOR_ONE_EPOCH_VAL = int(VALIDATION_NUM_OF_SAMPLES / BATCH_SIZE)  

openimages_generator = OpenImagesData(
  batch_size = BATCH_SIZE,
  len = int(NUM_OF_BATCHES_FOR_ONE_EPOCH / 15),
  num_of_classes = OpenImagesDataConstants.NUM_OF_CLASSES,
  total_number_of_samples=TOTAL_NUM_OF_SAMPLES
)

openimages_generator_val = OpenImagesData(
  batch_size = BATCH_SIZE,
  len = 15,
  num_of_classes = OpenImagesDataConstants.NUM_OF_CLASSES,
  total_number_of_samples=VALIDATION_NUM_OF_SAMPLES,
  table_name_for_image_urls='validation_images'
)

# checkpointer = ModelCheckpoint(filepath='./tmp/model.hdf5', verbose=1, save_best_only=False, period=0.01)

checkpointer = ModelSaver(every_n_minutes=118)

retrained_model.fit_generator(
  openimages_generator,
  epochs=10,
  callbacks=[checkpointer],
  use_multiprocessing=True,
  workers=10,
  validation_data=openimages_generator_val,
  validation_steps=int(VALIDATION_NUM_OF_SAMPLES / BATCH_SIZE / 10)
)
