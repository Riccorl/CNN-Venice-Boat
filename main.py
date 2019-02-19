import time

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

import VGGNet
import data_prep
import metrics
from utils import timer

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 30
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (128, 128, 3)
N_CLASSES = 6  # 6 = general, 19 = n < 24


def main():
    # check for gpu device
    print(tf.test.gpu_device_name())
    # print tf and keras version
    print(tf.VERSION)
    print(tf.keras.__version__)

    data, labels = data_prep.read_images("data/train", IMAGE_DIMS, general=True)
    print("Train Set loaded")
    data_test, labels_test = data_prep.read_images(
        "data/test", IMAGE_DIMS, general=True
    )
    print("Test Set loaded")

    lb = LabelBinarizer()
    X_train, y_train = data_prep.binarize(data, labels, lb)
    X_test, y_test = data_prep.binarize(data_test, labels_test, lb)
    print(X_train.shape)
    print(y_train.shape)

    model = VGGNet.vgg_net(N_CLASSES, IMAGE_DIMS)

    # Train the model
    start = time.time()
    history = VGGNet.fit(model, X_train, y_train, X_test, y_test, EPOCHS, BS)
    end = time.time()
    print("Training Time: " + timer(start, end))

    # Metrics
    metrics.plot_evaluation(history)
    y_pred = model.predict(X_test)
    classification_report = metrics.classification_report(y_test, y_pred, lb)
    print("Classification report : \n", classification_report)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix : \n", confusion_matrix)
    metrics.print_confusion_matrix(y_test, y_pred, lb)


if __name__ == "__main__":
    main()
