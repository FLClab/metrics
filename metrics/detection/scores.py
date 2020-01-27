
import numpy
import warnings

from scipy import spatial, optimize

class scores:
    """
    Class implementing common detection scores.
    """
    def __init__(self, truth, predicted, threshold, algorithm='nearest'):
        """
        Instantiates the `Scorer` object

        :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
        :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
        :param threshold: A distance threshold to consider a true positive detection (pixels)
        :param algorithm: A `string` specifying the assignation algorithm to use, can be
                          either 'nearest' or 'hungarian'
        """
        # Assign member variables
        self.truth = truth
        self.predicted = predicted
        self.threshold = threshold

        if algorithm is 'nearest':
            self.truth_couple, self.pred_couple = self._assign_nearest()
        elif algorithm is 'hungarian':
            self.truth_couple, self.pred_couple = self._assign_hungarian()

    def _assign_nearest(self):
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

    def _assign_hungarian(self):
        """
        Assigns each thruth detections to its nearest predicted detections.
        We consider a positive detections if it lies within a certain distance threshold.
        See `scipy.optimize.linear_sum_assignment` for more details about hungarian algo.

        :returns : The index of truth and predicted couples
        """

        truth_indices = numpy.arange(len(self.truth))
        pred_indices = numpy.arange(len(self.predicted))

        # Compute the pairwise distance matrice
        D = spatial.distance.cdist(self.truth, self.predicted, metric='euclidean')

        # We remove all points without neighbors in a radius of value `threshold`
        false_positives = numpy.sum(D < self.threshold, axis=0) == 0
        false_negatives = numpy.sum(D < self.threshold, axis=1) == 0

        # Remove all false positives and false negatives
        D = D[~false_negatives][:, ~false_positives]
        truth_indices = truth_indices[~false_negatives]
        pred_indices = pred_indices[~false_positives]

        # Apply the hungarian algorithm,
        # using log on the distance helps getting better matches
        # Because of the log, we need to ensure there is no Distance of 0
        D[D == 0] = 1e-6
        truth_couple, pred_couple = optimize.linear_sum_assignment(numpy.log(D))
        

        # Check if all distances are smaller than the threshold
        distances = D[truth_couple, pred_couple]
        truth_couple = truth_couple[distances < self.threshold]
        pred_couple = pred_couple[distances < self.threshold]

        truth_couple = truth_indices[truth_couple]
        pred_couple = pred_indices[pred_couple]

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


if __name__ == '__main__':
    dmatch = 5
    nb_pred = 90
    nb_gt = 100

    numpy.random.seed(2)

    truth = numpy.random.random_sample((nb_gt, 2)) * 100
    pred = numpy.random.random_sample((nb_pred, 2)) * 100

    s = scores(truth, pred, dmatch, algorithm='nearest')
    print(f'Truth index {s.truth_couple}')
    print(f'Pred index {s.pred_couple}')

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(truth[:, 0], truth[:, 1], marker='o')
    ax.scatter(pred[:, 0], pred[:, 1], marker='x')
    for i in range(len(s.truth_couple)):
        x1 = truth[s.truth_couple[i]]
        x2 = pred[s.pred_couple[i]]
        ax.text(*x1, str(s.truth_couple[i]))
        ax.text(*x2, str(s.pred_couple[i]))
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], 'r-')
        print(f'GT {s.truth_couple[i]}, pred {s.pred_couple[i]}, dist {numpy.linalg.norm(x1 - x2)}')

    s = scores(truth, pred, dmatch, algorithm='hungarian')
    print(f'Truth index {s.truth_couple}')
    print(f'Pred index {s.pred_couple}')

    ax = axes[1]
    ax.scatter(truth[:, 0], truth[:, 1], marker='o')
    ax.scatter(pred[:, 0], pred[:, 1], marker='x')
    for i in range(len(s.truth_couple)):
        x1 = truth[s.truth_couple[i]]
        x2 = pred[s.pred_couple[i]]
        ax.text(*x1, str(s.truth_couple[i]))
        ax.text(*x2, str(s.pred_couple[i]))
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], 'r-')
        print(f'GT {s.truth_couple[i]}, pred {s.pred_couple[i]}, dist {numpy.linalg.norm(x1 - x2)}')
    plt.show()
