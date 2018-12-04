import cv2
import glob
import os
import numpy as np


class FruitsDb:
    def __init__(self, dir, size=(50, 50), rotate=False,
                 cache_location='cache'):
        self.base_dir = dir
        self.train_dir = dir + os.sep + 'Training'
        self.test_dir = dir + os.sep + 'Test'
        self.cache = dir + os.sep + cache_location
        self.rotate = rotate
        self.size = size
        if not os.path.exists(self.cache):
            os.mkdir(self.cache)

    def get_training_data(self):
        if not self.rotate:
            image_cache = self.cache + os.sep + "training_images.npy"
            label_cache = self.cache + os.sep + "training_labels.npy"
        else:
            image_cache = self.cache + os.sep + "training_images_rotated.npy"
            label_cache = self.cache + os.sep + "training_labels_rotated.npy"
        try:
            print("Trying to load from cache...")
            images = np.load(image_cache)
            labels = np.load(label_cache)
            print("Loaded from cache!")
        except FileNotFoundError:
            print("No cached data!")
            images, labels = self.read_data(self.train_dir + os.sep + '*')
            np.save(image_cache, images)
            np.save(label_cache, labels)
        return images, labels

    def get_test_data(self):
        if not self.rotate:
            image_cache = self.cache + os.sep + "test_images.npy"
            label_cache = self.cache + os.sep + "test_labels.npy"
        else:
            image_cache = self.cache + os.sep + "test_images_rotated.npy"
            label_cache = self.cache + os.sep + "test_labels_rotated.npy"
        try:
            print("Trying to load from cache...")
            images = np.load(image_cache)
            labels = np.load(label_cache)
            print("Loaded from cache!")
        except FileNotFoundError:
            print("No cached data!")
            images, labels = self.read_data(self.train_dir + os.sep + '*')
            np.save(image_cache, images)
            np.save(label_cache, labels)
        return images, labels

    def read_data(self, dir):
        if self.rotate:
            rotation_matrices = [cv2.getRotationMatrix2D((self.size[0]/2,
                                                          self.size[1]/2),
                                                         angle, 1)
                                 for angle in range(15, 360, 15)]
        fruit_images = []
        labels = []
        for fruit_dir_path in glob.glob(dir):
            print('Reading ' + fruit_dir_path)
            fruit_label = fruit_dir_path.split(os.sep)[-1]

            for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                image = cv2.resize(image, self.size)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                fruit_images.append(image)
                labels.append(fruit_label)
                if self.rotate:
                    for matrix in rotation_matrices:
                        rotated_image = cv2.warpAffine(image, matrix, self.size)
                        fruit_images.append(np.asarray(rotated_image,
                                                       dtype=np.int8))
                        labels.append(fruit_label)

        fruit_images = np.vstack(fruit_images)

        # Normalize values between 0 and 1
        fruit_images = fruit_images / 255

        label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
        labels = np.array(labels)
        label_ids = np.array([label_to_id_dict[x] for x in labels])

        return fruit_images, label_ids

    def get_train_dir(self):
        return self.train_dir

    def get_test_dir(self):
        return self.test_dir
