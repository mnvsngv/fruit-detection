import os

import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np

from fruits_db import FruitsDb
from images.segmentation import watershed, selective_search
from models import dense, svm
from models.cnn import cnn

FIGSIZE = (50, 50)
NUM_FRUITS = 81

if __name__ == "__main__":
    # image_path = './data/test/fruits1.jpg'
    # img = cv.imread(image_path)
    # # resize image
    # newHeight = 500
    # newWidth = int(img.shape[1] * 200 / img.shape[0])
    # img = cv.resize(img, (newWidth, newHeight))
    # rects = selective_search(img)
    # model = cnn(img, rects)

    db_base_dir = '.' + os.sep + 'data' + os.sep + 'fruits-360'
    fruits_db = FruitsDb(db_base_dir, size=FIGSIZE, rotate=True)
    training_samples, training_labels = fruits_db.get_training_data()
    test_samples, test_labels = fruits_db.get_test_data()

    train_dense = True
    train_svm = False

    if train_dense:
        trained_model = dense.train_dense(0, FIGSIZE, NUM_FRUITS,
                                          training_samples, training_labels)

        loss, accuracy = dense.evaluate_model(trained_model, test_samples,
                                              test_labels)
        print("Loss: " + str(loss))
        print("Accuracy: " + str(accuracy * 100) + "%")

    if train_svm:
        trained_model = svm.train_svm(training_samples, training_labels)
        accuracy = svm.evaluate_model(trained_model, test_samples, test_labels)
        print("Accuracy: " + str(accuracy * 100) + "%")
