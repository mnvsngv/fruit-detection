import os

import cv2 as cv
import keras
import matplotlib.pyplot as plt
import numpy as np

from fruits_db import FruitsDb
from images.segmentation import watershed, selective_search
from models import dense, svm, convolutional
from models.cnn import cnn

FIGSIZE = (45, 45, 3)
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
    fruits_db = FruitsDb(db_base_dir, size=FIGSIZE, rotate=False)
    training_samples, training_labels = fruits_db.get_training_data()
    test_samples, test_labels = fruits_db.get_test_data()

    train_cnn = False
    train_dense = True
    train_svm = False

    if train_cnn:
        trained_model = convolutional.train_convolutional(0, FIGSIZE, NUM_FRUITS,
                                                          training_samples, training_labels)

        loss, accuracy = dense.evaluate_model(trained_model, test_samples,
                                              test_labels)
        print("Loss: " + str(loss))
        print("Accuracy: " + str(accuracy * 100) + "%")

    if train_dense:
        num_inputs = FIGSIZE[0] * FIGSIZE[1] * FIGSIZE[2]

        for i in range(0, 3):
            trained_model = dense.train_dense(i, num_inputs, NUM_FRUITS,
                                              training_samples, training_labels)

            loss, accuracy = dense.evaluate_model(trained_model, test_samples,
                                                  test_labels)
            print("Model " + str(i) + " results:")
            print("Loss: " + str(loss))
            print("Accuracy: " + str(accuracy * 100) + "%")


    if train_svm:
        trained_model = svm.train_svm(training_samples, training_labels)
        accuracy = svm.evaluate_model(trained_model, test_samples, test_labels)
        print("Accuracy: " + str(accuracy * 100) + "%")
