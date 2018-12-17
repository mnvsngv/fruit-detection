import os

import cv2 as cv
import joblib
from sklearn.svm import SVC

CACHE_DIR = "cache"


def train_svm(x_train, y_train, kernel):
    model_file = CACHE_DIR + os.sep + "svm_" + kernel + ".joblib"

    if os.path.exists(model_file):
        print("Loading model...")
        model = joblib.load(model_file)
        print("Loaded!")
    else:
        print("Preprocessing data...")
        x_train = [cv.cvtColor(img.astype('float32'),
                               cv.COLOR_BGR2GRAY).flatten()
                   for img in x_train]
        model = SVC(kernel=kernel, C=1.0)
        print("Training model...")
        model.fit(x_train, y_train)
        print("Done!")
        joblib.dump(model, model_file)
    return model


def evaluate_model(trained_model, x_test, y_test):
    print("Preprocessing data...")

    x_test = [cv.cvtColor(img.astype('float32'),
                          cv.COLOR_BGR2GRAY).flatten()
              for img in x_test]
    print("Predicting!")
    return trained_model.score(x_test, y_test)
