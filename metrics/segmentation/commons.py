
import numpy

from sklearn import metrics

def confusion_matrix(truth, predicted, normalized=True):
    """
    Computes the confusion matrix. By default the normalized confusion
    matrix is returned.

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :returns : A confusion matrix
    """
    if truth.ndim > 1:
        truth, predicted = truth.ravel(), predicted.ravel()
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    cm = metrics.confusion_matrix(truth, predicted)
    if normalized:
        cm = cm / cm.sum(axis=1)[:, numpy.newaxis]
    return cm

def iou(truth, predicted):
    """
    Computes the intersection over union

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :returns : Intersection over union
    """
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    intersection = (truth * predicted).sum()
    union = (truth + predicted).sum()
    return intersection / union

def dice(truth, predicted):
    """
    Computes the dice similarity coefficient

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :returns : Dice similarity coefficient
    """
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    intersection = (truth * predicted).sum()
    return 2 * intersection / (truth.sum() + predicted.sum())
