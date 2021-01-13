
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
        cm = cm / (cm.sum(axis=1)[:, numpy.newaxis] + 1e-12)
    return cm

def iou(truth, predicted):
    """
    Computes the intersection over union. If truth and predicted are both labeled
    images, a matrix is returned between each elements.

    :param truth: A 2D int `numpy.ndarray` of ground truth
    :param predicted: A 2D int `numpy.ndarray` of prediction

    :returns : A 2D `numpy.ndarray` of intersection over union
    """
    # Count objects
    true_objects = len(numpy.unique(truth))
    pred_objects = len(numpy.unique(predicted))

    # Compute intersection
    h = numpy.histogram2d(truth.flatten(), predicted.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]

    # Area of objects
    area_true = numpy.histogram(truth, bins=true_objects)[0]
    area_pred = numpy.histogram(predicted, bins=pred_objects)[0]

    # Calculate union
    area_true = numpy.expand_dims(area_true, -1)
    area_pred = numpy.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union

    return IOU

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
