# Import section

# import numpy
import numpy as np
# import tensorflow
import tensorflow as tf
# import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import random
import pickle
import cv2
import os

import time
# import os
import os
#import string
import string
# import regex
import re

# eager execution for debugging
# tf.enable_eager_execution()
# check for gpu device
print(tf.test.gpu_device_name())
# print tf and keras version
print(tf.VERSION)
print(tf.keras.__version__)

# timer function
def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

"""# Google Colab operations."""

# mount Drive
from google.colab import drive
drive.mount('/content/drive')

"""# Prepare the dataset"""

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 30
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (128, 128, 3)
N_CLASSES = 19 #  6 = general, 19 = n < 24

"""## General classes"""

classes = np.array([['People Transport'], ['General Transport'], ['Pleasure Craft'], ['Rowing Transport'], ['Public Utility'], ['Water']])
class_name = ['People Transport', 'General Transport', 'Pleasure Craft', 'Rowing Transport', 'Public Utility', 'Water']


# define the function blocks
def people_transport():
    return 'People Transport'

def general_transport():
    return 'General Transport'

def pleasure_craft():
    return 'Pleasure Craft'

def rowing_transport():
    return 'Rowing Transport'

def public_utility():
    return 'Public Utility'

def water():
    return 'Water'

# general classes
def general_classes(label):
    switch = {
        'Alilaguna': people_transport,
        'Lanciafino10m': people_transport,
        'Lanciafino10mBianca': people_transport,
        'Lanciafino10mMarrone': people_transport,
        'Lanciamaggioredi10mBianca': people_transport,
        'Lanciamaggioredi10mMarrone': people_transport,
        'MotoscafoACTV': people_transport,
        'VaporettoACTV': people_transport,
        'Motobarca': general_transport,
        'Mototopo': general_transport,
        'Motopontonerettangolare': general_transport,
        'Raccoltarifiuti': general_transport,
        'Barchino': pleasure_craft,
        'Cacciapesca': pleasure_craft,
        'Patanella': pleasure_craft,
        'Sanpierota': pleasure_craft,
        'Topa': pleasure_craft,
        'Gondola': rowing_transport,
        'Caorlina': rowing_transport,
        'Sandoloaremi': rowing_transport,
        'Polizia': public_utility,
        'Ambulanza': public_utility,
        'VigilidelFuoco': public_utility,
        'Water': water
    }
    return switch[label]()

"""## 19 classes"""

classes = {'Alilaguna',
        'Lanciafino10m',
        'Lanciafino10mBianca',
        'Lanciafino10mMarrone',
        'Lanciamaggioredi10mBianca',
        'MotoscafoACTV',
        'VaporettoACTV',
        'Motobarca',
        'Mototopo',
        'Motopontonerettangolare',
        'Raccoltarifiuti',
        'Barchino',
        'Patanella',
        'Topa',
        'Gondola',
        'Sandoloaremi',
        'Polizia',
        'Ambulanza',
        'Water'
}

# loop over the input images
def read_images(path):
    data = []
    labels = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                # extract the class label from the image path and update the
                # labels list
                label = subdir.split(os.path.sep)[-1]
                if label in classes:
                    labels.append(label)
                    # load the image, pre-process it, and store it in the data list
                    image = cv2.imread(os.path.join(subdir, file))
                    image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
                    image = img_to_array(image)
                    data.append(image)
         
    return data, labels

def data_numpy(x, y):
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(x, dtype="float") / 255.0
    labels = np.array(y)
    # binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    return data, labels

data, labels = read_images('mardct_classification_dataset/data/train')
print('Train Set loaded')
data_test, labels_test = read_images('mardct_classification_dataset/data/test_torch')
print('Test Set loaded')

# scale the raw pixel intensities to the range [0, 1]
x_train = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
# binarize the labels
lb = LabelBinarizer()
y_train = lb.fit_transform(labels)

#le = LabelEncoder()
#le.fit(np.unique(labels))
#labels = le.transform(labels)
#y_train = tf.keras.utils.to_categorical(labels)

print(x_train.shape)
print(y_train.shape)

# scale the raw pixel intensities to the range [0, 1]
x_test = np.array(data_test, dtype="float") / 255.0
labels_test = np.array(labels_test)
# binarize the labels
y_test = lb.fit_transform(labels_test)

#le.fit(np.unique(labels_test))
#labels_test = le.transform(labels_test)
#y_test = tf.keras.utils.to_categorical(labels_test)

print(x_test.shape)
print(y_test.shape)

print(lb.classes_)

#train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

img_gen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_img_gen = ImageDataGenerator()

"""# Small VGG16"""

model = tf.keras.Sequential()

model.add(Conv2D(32, (3, 3), input_shape=IMAGE_DIMS, padding='same',
           activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_CLASSES, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start = time.time()
history = model.fit_generator(
    test_img_gen.flow(x_train, y_train),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS,
    validation_data=test_img_gen.flow(x_test, y_test),
    validation_steps=len(x_test) // BS,
    #callbacks=[
        # Interrupt training if `val_loss` stops improving for over 3 epochs
    #    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
    #],
    verbose=1
)
end = time.time()
print('Training Time: ' + timer(start, end))

"""# **Pre-trained model**"""

from keras.applications import VGG16
 
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

vgg_conv.summary()

def bottleneck_features(X, model, bottleneck):
    for i in range(len(X)):
        img = np.take(X, i, axis=0)
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img)
        bottleneck[i] = feature.flatten()

train_bottleneck = np.empty([len(x_train), 4*4*512])
test_bottleneck = np.empty([len(x_test), 4*4*512])

bottleneck_features(x_train, vgg_conv, train_bottleneck)
bottleneck_features(x_test, vgg_conv, test_bottleneck)

model = tf.keras.Sequential()
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_CLASSES, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start = time.time()
history =  model.fit(train_bottleneck, y_train,
              epochs=EPOCHS,
              batch_size=BS,
              validation_data=(test_bottleneck, y_test),
              verbose=1
)
end = time.time()
print('Training Time: ' + timer(start, end))

"""# Metrics"""

# evaluate the model
scores = model.evaluate_generator(test_img_gen.flow(x_test, y_test),
                                  workers=4,
                                  verbose=1)
print("%s: %.2f" % (model.metrics_names[1], scores[1]))

# evaluate the model
scores = model.evaluate(test_bottleneck, y_test, workers=4, verbose=1)
print("%s: %.2f" % (model.metrics_names[1], scores[1]))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].legend(['loss', 'val_loss'], loc='upper right', frameon=True, facecolor='white', fontsize='large')


axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(history.history['acc'])
axes[1].plot(history.history['val_acc'])
axes[1].legend(['acc', 'val_acc'], loc='lower right', frameon=True, facecolor='white', fontsize='large')

plt.show()

from sklearn.metrics import classification_report

y_pred = model.predict(x_test)
print(classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), target_names=lb.classes_))

# pre trained
from sklearn.metrics import classification_report

y_pred = model.predict(test_bottleneck)
print(classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), target_names=lb.classes_))

from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), )
print('Confusion Matrix : \n', cm)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(test_bottleneck)
cm = confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), )
print('Confusion Matrix : \n', cm)

import seaborn as sns
import pandas as pd


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

class_names = lb.classes_
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
print_confusion_matrix(cnf_matrix, class_names)

y_pred = model.predict(x_test)
acc = sum([np.argmax(y_test[i])==np.argmax(y_pred[i]) for i in range(1682)])/1682
print(acc)