from PIL import Image
import numpy as np
from skimage.transform import resize


def load_image(img_path):
    img = Image.open(img_path)

    img.load()
    img_array = np.asarray(img, dtype='int32')

    return img_array


def image_downsample(img_array):
    arr = np.asarray(img_array)
    if arr.ndim == 3:
        arr = convert_image_to_grayscale(arr)

    arr = arr.astype(np.float32) / 255.0
    return resize(arr, (28, 28), anti_aliasing=True).astype(np.float32)


def convert_image_to_grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
