import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Dropout, Flatten, Dense


def vgg_net_trained(n_classes):
    model = tf.keras.Sequential()
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fit(model, X_train, y_train, X_test, y_test, input_shape, epochs, batch_size):
    train_bottleneck, test_bottleneck = bottleneck_features(input_shape, X_train, X_test)
    return model.fit(train_bottleneck, y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(test_bottleneck, y_test),
                     verbose=1
                     )


def bottleneck_features(input_shape, X_train, X_test):
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    train_bottleneck = get_bottleneck(X_train, vgg_conv)
    test_bottleneck = get_bottleneck(X_test, vgg_conv)
    return train_bottleneck, test_bottleneck


def get_bottleneck(X, model):
    bottleneck = np.empty([len(X), 4 * 4 * 512])
    for i in range(len(X)):
        img = np.take(X, i, axis=0)
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img)
        bottleneck[i] = feature.flatten()
    return bottleneck
