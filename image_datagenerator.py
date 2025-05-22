import numpy as np
import os
from os.path import isdir, join, isfile
import keras
from keras.preprocessing.image import img_to_array
from keras import backend as K
import cv2

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

class DirectoryDataGenerator(keras.utils.Sequence):
    def __init__(self, base_directories, augmentor=False, preprocessors=None,
                 batch_size=16, target_sizes=(224, 224), nb_channels=3, shuffle=False, verbose=True):
        self.base_directories = base_directories
        self.augmentor = augmentor
        self.preprocessors = preprocessors  # should be a function
        self.batch_size = batch_size
        self.target_sizes = target_sizes
        self.nb_channels = nb_channels
        self.shuffle = shuffle

        self.class_names = []
        files, labels = [], []

        for base_directory in base_directories:
            class_names = sorted([x for x in os.listdir(base_directory) if isdir(join(base_directory, x))])
            if self.class_names and len(class_names) != len(self.class_names[0]):
                raise Exception("All directories must contain the same number of classes.")

            self.class_names.append(class_names)

            for i, class_name in enumerate(class_names):
                class_dir = join(base_directory, class_name)
                for f in os.listdir(class_dir):
                    file_path = join(class_dir, f)
                    if isfile(file_path) and f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        files.append(file_path)
                        labels.append(i)

        self.nb_classes = len(self.class_names[0])
        self.files = files
        self.labels = labels
        self.nb_files = len(files)
        self.on_epoch_end()

        if verbose:
            print(f"Found {self.nb_files} images across {self.nb_classes} classes.")
            for i, class_list in enumerate(zip(*self.class_names)):
                print(f"Label {i}: {class_list}")

    def __len__(self):
        return int(np.floor(self.nb_files / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.files[k] for k in indexes]
        X = self._generate_X(batch_files)
        y = keras.utils.to_categorical([self.labels[k] for k in indexes], self.nb_classes)
        return X, y


    def on_epoch_end(self):
        self.indexes = np.arange(self.nb_files)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, batch_files):
        X = np.empty((self.batch_size, *self.target_sizes, self.nb_channels), dtype=K.floatx())

        for i, file_path in enumerate(batch_files):
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Image at {file_path} could not be read.")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

            if self.augmentor:
                img = cv2.resize(img, (256, 256))
                img = self.cv2_image_augmentation(img, theta=15, scale=0.15)
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)
                img = self.random_crop(img, self.target_sizes[0])
            else:
                img = cv2.resize(img, self.target_sizes)

            if self.preprocessors:
                img = self.preprocessors(img)

            X[i] = img

        return X

    def random_crop(self, image, crop_size):
        h, w = image.shape[:2]
        if h > crop_size and w > crop_size:
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)
            return image[top:top + crop_size, left:left + crop_size]
        return cv2.resize(image, (crop_size, crop_size))

    def cv2_image_augmentation(self, img, theta=20, tx=10, ty=10, scale=1.0):
        scale = np.random.uniform(1 - scale, 1 + scale) if scale != 1.0 else 1.0
        theta = np.random.uniform(-theta, theta) if theta != 0 else 0
        tx = np.random.uniform(-tx, tx) if tx != 0 else 0
        ty = np.random.uniform(-ty, ty) if ty != 0 else 0

        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, theta, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
