import os
import keras
import numpy as np
import cv2
import re
from scipy.misc import imread
import skimage.transform as transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from densenetpretrained.custom_layers import Scale

def process_openimages_multilabel_predictions(predictions):
  print("predictions shape: %s" % str(np.array(predictions).shape))
  prediction = predictions[0]

  matched_indices = []
  max_prob = -1
  max_prob_index = -1
  for label_idx, prob in enumerate(prediction):
    if prob > max_prob:
      max_prob = prob
      max_prob_index = label_idx
    # if prob > 0.1:
    #   matched_indices += [(label_idx, prob)]

  print("Matched index: %d " % max_prob_index)
  print("Matched index prob: %f " % max_prob)

  pass

def load_test_img(filepath):
  # img_numpy = mpimg.imread(filepath)
  im = Image.open(filepath)
  im = im.resize((224, 224))

  img_resized = np.array(im)
  # with open(file, mode = 'rb') as img_file:
  #   # bytes_img = await download_data_from_url(session, url)

  #   img_numpy = np.fromstring(img_file, np.uint8)

  #   # img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

  # img_resized = transform.resize(img_resized, (224, 224))
  img = img_resized.astype(np.uint8)
  return img


def perform_multilabel_classification_densenet(X):
  files_in_models = os.listdir('./checkpoints')
  files_in_models = [file for file in files_in_models if re.match('.*hdf5.*', file)]

  file_for_latest_model = sorted(files_in_models)[0]

  print("File for latest model %s" % file_for_latest_model)

  print("Loading model...")
  with keras.utils.CustomObjectScope({'Scale': Scale}):
    model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'checkpoints', file_for_latest_model))

  print("Predicting...")
  print(np.array([X]).shape)
  predictions = model.predict(np.array([X]))

  print("Processing predictions...")
  process_openimages_multilabel_predictions(predictions)

if __name__ == "__main__":
  X = load_test_img('./resources/sample.jpg')
  perform_multilabel_classification_densenet(X)
  # print(X)

  # fig = plt.figure()
  # plt.imshow(X)
  # plt.show()
