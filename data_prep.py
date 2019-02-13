import os

import cv2
import numpy as np
from tensorflow.python.keras.preprocessing.image import img_to_array

import classes


def binarize(X, y, lb):
    # scale the raw pixel intensities to the range [0, 1]
    X_bin = np.array(X, dtype="float") / 255.0
    y_labels = np.array(y)
    # binarize the labels
    y_bin = lb.fit_transform(y_labels)

    return X_bin, y_bin


def read_images(path, image_size, general=True):
    data = []
    labels = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                if general:
                    # load the image, pre-process it, and store it in the data list
                    image = cv2.imread(os.path.join(subdir, file))
                    image = cv2.resize(image, (image_size[0], image_size[1]))
                    image = img_to_array(image)
                    data.append(image)

                    # extract the class label from the image path and update the
                    # labels list
                    label = subdir.split(os.path.sep)[-1]
                    labels.append(classes.general_classes(label))
                else:
                    # extract the class label from the image path and update the
                    # labels list
                    label = subdir.split(os.path.sep)[-1]
                    if label in classes.specific_classes:
                        labels.append(label)
                        # load the image, pre-process it, and store it in the data list
                        image = cv2.imread(os.path.join(subdir, file))
                        image = cv2.resize(image, (image_size[0], image_size[1]))
                        image = img_to_array(image)
                        data.append(image)

    return data, labels
