import os
import keras
import numpy as np
import re
from PIL import Image
from densenetpretrained.custom_layers import Scale
import argparse
from utils.get_last_checkpoint_from_dir import get_last_checkpoint_from_dir

def process_openimages_multilabel_predictions(predictions):
  print("predictions shape: %s" % str(np.array(predictions).shape))
  prediction = predictions[0]

  matched_indices = []
  max_prob = -1
  max_prob_index = -1
  sum_of_probs = 0

  label_indices_matched = []
  prob_threshold = 0.2
  for label_idx, prob in enumerate(prediction):
    if prob > prob_threshold:
      label_indices_matched += [label_idx]
    if prob > max_prob:
      max_prob = prob
      max_prob_index = label_idx
    # if prob > 0.1:
    #   matched_indices += [(label_idx, prob)]

    sum_of_probs += prob

  print("Matched index: %d " % max_prob_index)
  print("Matched labels: %s " % str(label_indices_matched))
  print("Matched index prob: %f " % max_prob)
  print("Sum of probs: %f" % sum_of_probs)

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


def perform_multilabel_classification_densenet(X, checkpoint_dir):
  file_for_latest_model = get_last_checkpoint_from_dir(checkpoint_dir)

  print("File for latest model %s" % file_for_latest_model)

  print("Loading model...")
  with keras.utils.CustomObjectScope({'Scale': Scale}):
    model_path = os.path.join(os.path.dirname(__file__), checkpoint_dir, file_for_latest_model)

    print(model_path)
    model = keras.models.load_model(model_path)

  print("Predicting...")
  print(np.array([X]).shape)
  predictions = model.predict(np.array([X]))

  print("Processing predictions...")
  process_openimages_multilabel_predictions(predictions)

# if __name__ == "__main__":
#   X = load_test_img('./resources/sample.jpg')
#   perform_multilabel_classification_densenet(X)
#   # print(X)

  # fig = plt.figure()
  # plt.imshow(X)
  # plt.show()


def main(args):
  X = load_test_img(args.path_to_test_image)
  perform_multilabel_classification_densenet(X, args.checkpoint_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Retrain pretrained vision network for openimages.')

  parser.add_argument('--checkpoint_dir', type=str, choices=['./checkpoints/inceptionV3/2000', './checkpoints'], required=True,
                      help='Directory containing checkpoint for model to restore.')
  parser.add_argument('--path_to_test_image', type=str, required=True,
                      help='Path to the image which we want to annotate..')
  args = parser.parse_args()

  main(args)
