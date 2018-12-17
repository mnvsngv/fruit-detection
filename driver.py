import os

import cv2 as cv

from fruits_db import FruitsDb
from images.detection import object_detection
from images.segmentation import selective_search
from models import dense, svm, convolutional

FIGSIZE = (45, 45, 3)
NUM_FRUITS = 81

if __name__ == "__main__":

    # Create the corpus directory object, which will allow us to access
    # training and test data more easily
    db_base_dir = '.' + os.sep + 'data' + os.sep + 'fruits-360'
    fruits_db = FruitsDb(db_base_dir, size=FIGSIZE, rotate=False)
    training_samples, training_labels = fruits_db.get_training_data()
    test_samples, test_labels = fruits_db.get_test_data()

    # Switches to determine which kind of model to train and test. When
    # detection is True, train_cnn *HAS* to be True. In this case,
    # the evaluation of the CNN is also skipped.
    train_cnn = True
    train_dense = False
    train_svm = False
    detection = True

    if train_cnn:
        print("CNNs")

        # Since we have 3 models, we train 3
        for i in range(0, 3):
            trained_model = convolutional.train_convolutional(i, FIGSIZE,
                                                              NUM_FRUITS,
                                                              training_samples,
                                                              training_labels,
                                                              verbose=0)

            print("Testing!")
            if not detection:
                loss, accuracy = dense.evaluate_model(trained_model,
                                                      test_samples, test_labels)
                print("Model " + str(i) + " results:")
                print("Loss: " + str(loss))
                print("Accuracy: " + str(accuracy * 100) + "%")

    if train_dense:
        print("DNNs")
        num_inputs = FIGSIZE[0] * FIGSIZE[1] * FIGSIZE[2]

        # Since we have 3 models, we train 3
        for i in range(0, 3):
            trained_model = dense.train_dense(i, num_inputs, NUM_FRUITS,
                                              training_samples, training_labels,
                                              verbose=1)

            loss, accuracy = dense.evaluate_model(trained_model, test_samples,
                                                  test_labels)
            print("Model " + str(i) + " results:")
            print("Loss: " + str(loss))
            print("Accuracy: " + str(accuracy * 100) + "%")

    if train_svm:
        print("SVMs")

        # We train 3 different types of SVMs by changing their kernels
        for kernel_type in ["linear", "rbf", "poly"]:
            trained_model = svm.train_svm(training_samples, training_labels,
                                          kernel_type)
            accuracy = svm.evaluate_model(trained_model, test_samples,
                                          test_labels)
            print("Kernel: " + kernel_type + "; Accuracy: "
                  + str(accuracy * 100) + "%")
            del trained_model

    if detection:
        image_path = os.path.join('.', 'data', 'test', 'fruits1.jpg')
        img = cv.imread(image_path)
        # Resize image
        newHeight = 500
        newWidth = int(img.shape[1] * 500 / img.shape[0])
        img = cv.resize(img, (newWidth, newHeight))
        rects = selective_search(img)

        object_detection(img, rects, trained_model, fruits_db, FIGSIZE)
