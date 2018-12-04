import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
import glob
import cv2

import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop, SGD

from keras import backend as K


def cnn(img, rects):
    # try:
    #     print("Loading data...")
    #     model_cnn = keras.models.load_model("trained_cnn")
    #     id_to_labels = np.load("labels.npy").item()
    #     print(id_to_labels)
    # except OSError:
    print("Training!")
    # Read Training dataset
    fruit_images = []
    labels = []
    for fruit_dir_path in glob.glob('.' + os.sep + 'data' + os.sep +
                                    'fruits-360' + os.sep + 'Training' +
                                    os.sep + '*'):
        print('in')
        fruit_label = fruit_dir_path.split(os.sep)[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            fruit_images.append(image)
            labels.append(fruit_label)
    fruit_images = np.array(fruit_images)
    labels = np.array(labels)
    print(labels)

    # key- value for id and label
    label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
    id_to_labels = {i: v for i, v in enumerate(np.unique(labels))}
    np.save("labels", label_to_id_dict)

    label_ids = np.array([label_to_id_dict[x] for x in labels])

    print(label_ids)

    print(fruit_images.shape)
    print(label_ids.shape)
    print(labels.shape)

    # Read Testing dataset
    test_fruit_images = []
    test_labels = []
    dir = '.' + os.sep + 'data' + os.sep + \
          'fruits-360' + os.sep + 'Test' + os.sep + '*'
    for fruit_dir_path in glob.glob(dir):
        fruit_label = fruit_dir_path.split(os.sep)[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            test_fruit_images.append(image)
            test_labels.append(fruit_label)
    test_fruit_images = np.array(test_fruit_images)
    test_labels = np.array(test_labels)

    validation_label_ids = np.array([label_to_id_dict[x] for x in test_labels])
    print(test_labels)

    X_train, X_test = fruit_images, test_fruit_images
    Y_train, Y_test = label_ids, validation_label_ids

    # Normalize color values to between 0 and 1
    X_train = X_train / 255
    X_test = X_test / 255

    # One Hot Encode the Output
    Y_train = keras.utils.to_categorical(Y_train, 81)
    Y_test = keras.utils.to_categorical(Y_test, 81)

    # CNN model
    model_cnn = Sequential()
    # First convolutional layer, note the specification of shape
    model_cnn.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(45, 45, 3)))
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(81, activation='softmax'))

    model_cnn.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    model_cnn.fit(X_train, Y_train,
                  batch_size=128,
                  epochs=8,
                  verbose=1,
                  validation_data=(X_test, Y_test))
    score = model_cnn.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # keras.models.save_model(model_cnn, "trained_cnn")

    images = [cv2.resize(img[y:y + h, x:x + w], (45, 45)) for x, y, w, h in rects]
    results = model_cnn.predict(np.array(images))

    objects, types = np.where(results > 0.9)
    for i, object in enumerate(objects):
        x, y, w, h = rects[object]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.putText(img, id_to_labels[types[i]], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0))
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    return results
