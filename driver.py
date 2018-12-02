import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np

from images.segmentation import watershed, selective_search
from networks.cnn import cnn

if __name__ == "__main__":
    image_path = './data/test/fruits1.jpg'
    img = cv.imread(image_path)
    rects = selective_search(image_path)
    model = cnn(img, rects)
