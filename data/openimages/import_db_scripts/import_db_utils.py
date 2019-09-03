import sys
import aiohttp
from utils.preprocess_image import resize_img


async def download_data_from_url(session, url):
    try:
        async with session.get(url) as response:
            if response.status != 200 and response.status != '200':
                print(f'response.status: {response.status}')
                return None
            response_text = await response.read()

            return response_text
    except Exception as exc:
        print("Error in download_data_from_url(session, url): " + str(exc), file=sys.stderr)


async def async_download_image(image_url, target_img_size, *, letterbox_image=False):
    try:
        async with aiohttp.ClientSession() as session:
            bytes_img = await download_data_from_url(session, image_url)
            if bytes_img is None:
                return None

            img = resize_img(bytes_img, target_img_size, letterbox_image)

            return img
    except Exception as e:
        print("Error in download_image(url, id):  " + str(e), file = sys.stderr)
        print(f'bytes_img: {bytes_img}')
        raise Exception(f'Failed to download: {str(e)}')


