import os
import keras
import numpy as np
import re
from PIL import Image
import argparse

from utils.get_last_checkpoint_from_dir import get_last_checkpoint_from_dir
from data.openimages import db
from common.argparse_types import str2bool
from keras_custom.metrics.f1_score import f1
from utils.preprocess_image import preprocess_image_bytes
from keras_custom.losses.masked_binary_crossentropy import make_masked_binary_cross_entropy
from data.openimages.constants import Constants


def process_openimages_multilabel_predictions(predictions, use_trainable_labels):
    print("predictions shape: %s" % str(np.array(predictions).shape))
    prediction = predictions[0]

    matched_indices = []
    max_prob = -1
    max_prob_index = -1
    sum_of_probs = 0

    label_indices_matched = []
    prob_threshold = 0.2
    predictions = []

    for label_idx, prob in enumerate(prediction):
        if prob > prob_threshold:
            label_indices_matched += [label_idx]
        if prob > max_prob:
            max_prob = prob
            max_prob_index = label_idx

        predictions += [(prob, label_idx)]
        # if prob > 0.1:
        #   matched_indices += [(label_idx, prob)]

        sum_of_probs += prob

    predictions = sorted(predictions, key=lambda prediction: prediction[0])
    most_probable_predictions = predictions[-args.most_probable_predictions:]

    print("Matched index: %d " % max_prob_index)
    print("Matched labels: %s " % str(label_indices_matched))
    print("Matched index prob: %f " % max_prob)
    print("Sum of probs: %f" % sum_of_probs)

    print("Most probable predictions: ")
    print(most_probable_predictions)

    for prob, label_idx in most_probable_predictions:
        label_idx = label_idx + 1

        if use_trainable_labels:
            labels = db.get_trainable_label_by_ids([label_idx])
        else:
            labels = db.get_labels_by_ids([label_idx])
        predicted_label = labels[0]

        print(
            f'With probability {prob} network detected label: {predicted_label[1]} - (id = {predicted_label[0]})')


def load_test_img(filepath):
    im = Image.open(filepath)
    im = im.resize((225, 225))

    img_resized = np.array(im)
    img = img_resized.astype(np.uint8)
    return img


def perform_multilabel_classification(X, checkpoint_dir, file_for_latest_model, use_trainable_labels):
    # file_for_latest_model = get_last_checkpoint_from_dir(checkpoint_dir)
    print("File for latest model %s" % file_for_latest_model)

    print("Loading model...")
    model_path = os.path.join(os.path.dirname(__file__), checkpoint_dir, file_for_latest_model)

    print(model_path)
    model = keras.models.load_model(model_path, {
        'f1': f1,
        'masked_binary_cross_entropy': make_masked_binary_cross_entropy(Constants.MASK_VALUE_MISSING_LABEL)
    })

    print("Predicting...")
    print(np.array([X]).shape)
    predictions = model.predict(preprocess_image_bytes(np.array([X])))

    print("Processing predictions...")
    process_openimages_multilabel_predictions(predictions, use_trainable_labels)


def main(args):
    X = load_test_img(args.path_to_test_image)
    perform_multilabel_classification(
        X, args.checkpoint_dir, args.file_for_latest_model, args.use_trainable_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test infering using trained vision network for openimages.')

    parser.add_argument('--checkpoint_dir',
                        type=str,
                        required=False,
                        default='./checkpoints/multilabel_classification/inceptionV3/1',
                        help='Directory containing checkpoint for model to restore.')

    parser.add_argument('--file_for_latest_model',
                        type=str,
                        required=True,
                        help='File containing checkpoint for model to restore.')

    parser.add_argument('--path_to_test_image',
                        type=str,
                        required=False,
                        default='./resources/horse.jpeg',
                        help='Path to the image which we want to annotate..')

    parser.add_argument('--use_trainable_labels',
                        type=str2bool,
                        required=False,
                        default=True,
                        help='Flag whether to use trainable labels (or all labels).')

    parser.add_argument('--most_probable_predictions',
                        type=int,
                        required=False,
                        default=10,
                        help='How many most probably detections to output.')

    args = parser.parse_args()

    main(args)
