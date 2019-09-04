
import numpy

from scipy import spatial, optimize

__all__ = [
    "precision", "recall", "f1_score", "jaccard"
]

def precision(truth, predicted, threshold=5):
    """
    Computes the precision between the truth and the predictions. It is considered
    a true positive if it lies inside a certain distance threshold. We solve a
    linear sum assigment problem to associate each detections

    :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
    :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
    :param threshold: A distance threshold to consider a true positive detection (pixels)

    :returns : The precision
    """
    distances = spatial.distance.cdist(truth, predicted)
    row_ind, col_ind = optimize.linear_sum_assignment(distances)
    assignment = distances[row_ind, col_ind]
    true_positive = (assignment <= threshold).sum()
    false_positive = predicted.shape[0] - true_positive
    return true_positive / (true_positive + false_positive)

def recall(truth, predicted, threshold=5):
    """
    Computes the recall between the truth and the predictions. It is considered
    a true positive if it lies inside a certain distance threshold. We solve a
    linear sum assigment problem to associate each detections

    :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
    :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
    :param threshold: A distance threshold to consider a true positive detection (pixels)

    :returns : The recall
    """
    distances = spatial.distance.cdist(truth, predicted)
    row_ind, col_ind = optimize.linear_sum_assignment(distances)
    assignment = distances[row_ind, col_ind]
    true_positive = (assignment <= threshold).sum()
    false_negative = truth.shape[0] - true_positive
    return true_positive / (true_positive + false_negative)

def f1_score(truth, predicted, threshold=5):
    """
    Computes the F1-score between the truth and the predictions. We consider a
    true positive detection if it lies within a certain distance threshold. We
    solve a linear sum assigment problem to associate each detections

    :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
    :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
    :param threshold: A distance threshold to consider a true positive detection (pixels)

    :returns : The F1-score
    """
    prec = precision(truth, predicted, threshold=threshold)
    rec = recall(truth, predicted, threshold=threshold)
    if (prec == 0) and (rec == 0):
        return 0
    return 2 * (prec * rec) / (prec + rec)

def jaccard(truth, predicted, threshold=5):
    """
    Computes the Jaccard Index between the truth and the predictions. We consider a
    true positive detection if it lies within a certain distance threshold. We solve
    a linear sum assigment problem to associate each detections

    :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
    :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
    :param threshold: A distance threshold to consider a true positive detection (pixels)
    """
    f1 = f1_score(truth, predicted, threshold=threshold)
    return f1 / (2 - f1)
