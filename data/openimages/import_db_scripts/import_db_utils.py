import aiohttp
import aiosqlite
import cv2
import numpy as np
import sys
import aiohttp

async def download_data_from_url(session, url):
  try:
    async with session.get(url) as response:
      response_text = await response.read()

      return response_text
  except Exception as exc:
    print("Error in download_data_from_url(session, url): " + str(exc), file=sys.stderr)


async def async_download_image(image_url, target_img_size):
  try:
    async with aiohttp.ClientSession() as session:
      bytes_img = await download_data_from_url(session, image_url)

      img_numpy = np.fromstring(bytes_img, np.uint8)

      img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

      img_resized = cv2.resize(img_cv2, target_img_size)
      img = img_resized.astype(np.uint8)

      return img
  except Exception as e:
    print("Error in download_image(url, id):  " + str(e), file = sys.stderr)
    raise Exception(f'Failed to download: {str(e)}')


