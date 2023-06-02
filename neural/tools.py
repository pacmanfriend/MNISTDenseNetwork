import numpy as np
import cv2
from scipy import ndimage
import math


def load_image_cv(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    gray = cv2.resize(255 - gray, (28, 28))

    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rowsPadding, colsPadding), mode="constant")

    shiftx, shifty = get_best_shift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    return gray


def get_best_shift(img):
    if np.sum(img) == 0:
        return 0, 0
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return 0, 0
    shiftx = int(np.round(cols / 2.0 - cx))
    shifty = int(np.round(rows / 2.0 - cy))
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted
