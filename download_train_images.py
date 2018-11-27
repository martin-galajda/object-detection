import re
import os
import io
import requests
# from PIL import Image
from urllib.request import urlretrieve
import cv2
import numpy as np
# import matplotlib.pyplot as plt

def download_image(url, target_local_file):
  # data = requests.get(url).content
  # img = Image.open(io.BytesIO(data))
  # print(img)

  # urlretrieve(url, target_local_file)
  data = requests.get(url).content
  bytes_img = io.BytesIO(data)
  bytes_img = bytes_img.read()
  img_numpy = np.fromstring(bytes_img, np.uint8)

  img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

  img_resized = cv2.resize(img_cv2, (224, 224))

  return img_resized.astype(np.float32)

def download_images_from_tsv(path_to_tsv):
  path_to_tsv_tokens = path_to_tsv.split('/')
  dir_name = re.sub('.tsv', '', path_to_tsv_tokens[len(path_to_tsv_tokens) - 1])

  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  images = []

  with open(path_to_tsv) as tsv_file:
    line_idx  = 0

    for line in tsv_file:
      if line_idx == 0:
        line_idx += 1
        continue
      line_idx += 1

      cols = line.split('\t')
      image_url, image_bytes, image_id = cols

      ext = image_url.split('.')[len(image_url.split('.')) - 1]

      print("Downloading %i-th image." % (line_idx - 1))
      images += [{
        'id': image_id,
        'pixels': download_image(image_url, dir_name + '/' + image_id),
        'ext': ext
      }]
      print("Downloaded %i-th image." % (line_idx - 1))
      print("")


  return images

def is_tsv_file(filename):
  return re.search(r'.tsv', filename) != None


def download_images():
  images = []
  downloaded_bytes = 0

  for root, dirs, files in os.walk("."):
    for filename in files:
      if is_tsv_file(filename):
        images = download_images_from_tsv(root + '/' + filename)
  return images
