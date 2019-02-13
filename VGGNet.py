import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def vgg_net(n_classes, input_shape):
    model = tf.keras.Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', ))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', ))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

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


def fit(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    img_gen = ImageDataGenerator()

    return model.fit_generator(
        img_gen.flow(X_train, y_train),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=img_gen.flow(X_test, y_test),
        validation_steps=len(X_test) // batch_size,
        verbose=1
    )


def evaluate(model, X_train, y_train):
    img_gen = ImageDataGenerator()
    # evaluate the model
    scores = model.evaluate_generator(img_gen.flow(X_train, y_train),
                                      workers=4,
                                      verbose=1)
    print("%s: %.2f" % (model.metrics_names[1], scores[1]))
