
import numpy
import warnings

from scipy import spatial, optimize

class scores:
    """
    Class implementing common detection scores.
    """
    def __init__(self, truth, predicted, threshold):
        """
        Instantiates the `Scorer` object

        :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
        :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
        :param threshold: A distance threshold to consider a true positive detection (pixels)
        """
        # Assign member variables
        self.truth = truth
        self.predicted = predicted
        self.threshold = threshold

        self.truth_couple, self.pred_couple = self._assign()

    def _assign(self):
        """
        Assigns each truth detections to its nearest predicted detections. We consider
        a positive detections if it lies within a certain distance threshold.

        :returns : The index of truth and predicted couples
        """
        distances = spatial.distance.cdist(self.truth, self.predicted).ravel()
        dist_sorted = numpy.argsort(distances)
        truth_couple, pred_couple = [], []
        for arg in dist_sorted:
            if distances[arg] > self.threshold:
                break
            where = (arg // self.predicted.shape[0], arg - arg // self.predicted.shape[0] * self.predicted.shape[0])
            if (where[0] in truth_couple) or (where[1] in pred_couple):
                continue
            truth_couple.append(where[0])
            pred_couple.append(where[1])
        return truth_couple, pred_couple

    @property
    def tpr(self):
        """
        Computes the true positive rate between the truth and the predictions

        :returns : A true positive rate score
        """
        return self.recall

    @property
    def fnr(self):
        """
        Computes the false negative rate between the truth and the predictions

        :returns : A false negative rate score
        """
        true_positive = len(self.pred_couple)
        false_positive = len(self.predicted) - true_positive
        false_negative = len(self.truth) - true_positive
        return false_negative / (true_positive + false_positive)

    @property
    def fpr(self):
        """
        Computes the false positive rate between the truth and the predictions
        NOTE. In the case of truth detections there are no true negatives

        :returns : A false positive rate
        """
        warnings.warn("Using the false positive rate as a score metric in the case truth predictions does not make sense as there are no true negative labels.",
                        category=UserWarning)
        true_positive = len(self.pred_couple)
        false_positive = len(self.predicted) - true_positive
        false_negative = len(self.truth) - true_positive
        return false_positive / false_negative

    @property
    def accuracy(self):
        """
        Computes the accuracy between the truth and the predictions

        :returns : An accuracy score
        """
        return len(self.pred_couple) / len(self.truth)

    @property
    def precision(self):
        """
        Computes the precision between the truth and the predictions.

        :returns : A precision score
        """
        true_positive = len(self.pred_couple)
        false_positive = len(self.predicted) - true_positive
        return true_positive / (true_positive + false_positive)

    @property
    def recall(self):
        """
        Computes the recall between the truth and the predictions.

        :returns : A recall score
        """
        true_positive = len(self.pred_couple)
        false_negative = len(self.truth) - true_positive
        return true_positive / (true_positive + false_negative)

    @property
    def f1_score(self):
        """
        Computes the F1-score between the truth and the predictions.

        :returns : A F1-score
        """
        prec = self.precision
        rec = self.recall
        if (prec == 0) and (rec == 0):
            return 0
        return 2 * (prec * rec) / (prec + rec)

    @property
    def jaccard(self):
        """
        Computes the Jaccard Index between the truth and the predictions.

        :returns : A Jaccard index score
        """
        f1 = self.f1_score
        return f1 / (2 - f1)
