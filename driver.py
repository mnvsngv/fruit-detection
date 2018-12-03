import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np

from images.segmentation import watershed, selective_search
from networks.cnn import cnn

if __name__ == "__main__":
    image_path = './data/test/fruits3.jpg'
    img = cv.imread(image_path)
    # resize image
    newHeight = 500
    newWidth = int(img.shape[1] * 200 / img.shape[0])
    img = cv.resize(img, (newWidth, newHeight))
    rects = selective_search(img)
    model = cnn(img, rects)
