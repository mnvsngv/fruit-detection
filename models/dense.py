import os

import keras
from keras import regularizers, metrics
from keras.layers import Dense, Flatten
from keras.models import Sequential

CACHE_DIR = "cache" + os.sep


def get_model(model_num, num_inputs, num_outputs):
    # List of models
    model_dictionary = {
        0: [
            Flatten(),
            Dense(num_inputs, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(num_outputs, activation='softmax',
                  kernel_regularizer=regularizers.l2()),
        ],
        1: [
            Flatten(),
            Dense(num_inputs, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(num_outputs, activation='softmax',
                  kernel_regularizer=regularizers.l2()),
        ],
        2: [
            Flatten(),
            Dense(num_inputs, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(90, activation='relu', kernel_regularizer=regularizers.l2()),
            Dense(num_outputs, activation='softmax',
                  kernel_regularizer=regularizers.l2()),
        ],
    }
    model = Sequential()
    for layer in model_dictionary[model_num]:
        model.add(layer)

    model.compile(optimizer="Adam", loss="categorical_crossentropy",
                  metrics=[metrics.categorical_accuracy])

    return model


def train_dense(model_num, num_inputs, num_outputs, x_train, y_train,
                epochs=10, batch_size=128, verbose=0):
    try:
        print("Loading model...")
        trained_model = keras.models.load_model(CACHE_DIR +
                                                os.sep + "trained_dense_" +
                                                str(model_num))
    except OSError:
        print("Training!")
        # One Hot Encode the Output
        y_train = keras.utils.to_categorical(y_train)

        trained_model = get_model(model_num, num_inputs, num_outputs)
        trained_model.fit(x_train, y_train, batch_size=batch_size,
                          epochs=epochs, verbose=verbose)

        keras.models.save_model(trained_model, CACHE_DIR +
                                os.sep + "trained_dense_" + str(model_num))

    return trained_model


def evaluate_model(trained_model, x_test, y_test):
    y_test = keras.utils.to_categorical(y_test)
    score = trained_model.evaluate(x_test, y_test, verbose=0)
    return score
