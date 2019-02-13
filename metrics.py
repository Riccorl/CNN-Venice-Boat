import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics


def plot_evaluation(history):
    """Plot the value of Accuracy and Loss for each epoch"""
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


def classification_report(y_test, y_pred, lb):
    """Return the classification report"""
    return metrics.classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), target_names=lb.classes_)


def confusion_matrix(y_test, y_pred):
    """Return the confusion matrix"""
    return metrics.confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), )


def print_confusion_matrix(y_test, y_pred, lb):
    class_names = lb.classes_
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
    get_confusion_matrix(cnf_matrix, class_names)


def get_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
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
