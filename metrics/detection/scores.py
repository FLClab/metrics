
import numpy
import warnings

from scipy import spatial, optimize
from skimage import measure
from matplotlib import pyplot

from metrics import commons

class DetectionError:
    """
    Class implementing common detection scores
    """
    def __init__(self, truth, predicted, algorithm='hungarian', **kwargs):
        """
        Instantiates the `CentroidDetectionError` object

        :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
        :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
        :param algorithm: A `string` specifying the assignation algorithm to use, can be
                          either 'nearest' or 'hungarian'
        """
        # Assign member variables
        self.truth = truth
        self.predicted = predicted
        self.algorithm = algorithm

        # Assign kwargs variables as member variables
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Verifies if attribute exists, else defaults in hungarian
        try:
            self.assign = getattr(self, f"_assign_{algorithm}")
        except AttributeError:
            warnings.warn(f"The chosen algorithm `{algorithm}` does not exist. Defaults in `hungarian`.", category=UserWarning)
            self.assign = getattr(self, "_assign_hungarian")

        # Computes the cost matrix
        self.compute_cost_matrix()

    def _assign_nearest(self, threshold):
        """
        Assigns each truth detections to its nearest predicted detections. We consider
        a positive detections if it lies within a certain distance threshold.

        :returns : The index of truth and predicted couples
        """
        raveled_cost_matrix = self.cost_matrix.ravel()
        dist_sorted = numpy.argsort(raveled_cost_matrix)
        truth_couple, pred_couple = [], []
        for arg in dist_sorted:
            if raveled_cost_matrix[arg] > threshold:
                break
            where = (arg // self.cost_matrix.shape[1], arg - arg // self.cost_matrix.shape[1] * self.cost_matrix.shape[1])
            if (where[0] in truth_couple) or (where[1] in pred_couple):
                continue
            truth_couple.append(where[0])
            pred_couple.append(where[1])
        return truth_couple, pred_couple

    def _assign_hungarian(self, threshold, maximize=False):
        """
        Assigns each truth detections to its nearest predicted detections. We consider
        a positive detections if it lies within a certain distance threshold. The
        assignement uses the hungarian algorithm.
        See `scipy.optimize.linear_sum_assignment` for more details about hungarian algo.

        :param threshold: A `float` of threshold to apply
        :param maximize: (optional) Whether to maximize the assignement

        :returns : A `numpy.ndarray` of indices of truth
                   A `numpy.ndarray` of indices of pred
        """
        truth_indices = numpy.arange(self.cost_matrix.shape[0])
        pred_indices = numpy.arange(self.cost_matrix.shape[1])

        # We remove all points without neighbors in a radius of value `threshold`
        if maximize:
            false_positives = numpy.sum(self.cost_matrix >= threshold, axis=0) == 0
            false_negatives = numpy.sum(self.cost_matrix >= threshold, axis=1) == 0
        else:
            false_positives = numpy.sum(self.cost_matrix < threshold, axis=0) == 0
            false_negatives = numpy.sum(self.cost_matrix < threshold, axis=1) == 0

        # Remove all false positives and false negatives
        cost = self.cost_matrix[~false_negatives][:, ~false_positives]
        truth_indices = truth_indices[~false_negatives]
        pred_indices = pred_indices[~false_positives]

        # Apply the hungarian algorithm,
        # using log on the distance helps getting better matches
        # Because of the log, we need to ensure there is no Distance of 0
        if maximize:
            truth_couple, pred_couple = optimize.linear_sum_assignment(cost, maximize=maximize)
        else:
            truth_couple, pred_couple = optimize.linear_sum_assignment(numpy.log(cost + 1e-6), maximize=maximize)

            # Check if all distances are smaller than the threshold
            distances = cost[truth_couple, pred_couple]
            truth_couple = truth_couple[distances < threshold]
            pred_couple = pred_couple[distances < threshold]

        truth_couple = truth_indices[truth_couple]
        pred_couple = pred_indices[pred_couple]

        return truth_couple, pred_couple

    # def _assign_hungarian(self):
    #     """
    #     Assigns each thruth detections to its nearest predicted detections.
    #     We consider a positive detections if it lies within a certain distance threshold.
    #     See `scipy.optimize.linear_sum_assignment` for more details about hungarian algo.
    #
    #     :returns : The index of truth and predicted couples
    #     """
    #
    #     truth_indices = numpy.arange(len(self.truth))
    #     pred_indices = numpy.arange(len(self.predicted))
    #
    #     # Compute the pairwise distance matrix
    #     D = spatial.distance.cdist(self.truth, self.predicted, metric='euclidean')
    #
    #     # We remove all points without neighbors in a radius of value `threshold`
    #     false_positives = numpy.sum(D < self.threshold, axis=0) == 0
    #     false_negatives = numpy.sum(D < self.threshold, axis=1) == 0
    #
    #     # Remove all false positives and false negatives
    #     D = D[~false_negatives][:, ~false_positives]
    #     truth_indices = truth_indices[~false_negatives]
    #     pred_indices = pred_indices[~false_positives]
    #
    #     # Apply the hungarian algorithm,
    #     # using log on the distance helps getting better matches
    #     # Because of the log, we need to ensure there is no Distance of 0
    #     truth_couple, pred_couple = optimize.linear_sum_assignment(numpy.log(D + 1e-6))
    #
    #
    #     # Check if all distances are smaller than the threshold
    #     distances = D[truth_couple, pred_couple]
    #     truth_couple = truth_couple[distances < self.threshold]
    #     pred_couple = pred_couple[distances < self.threshold]
    #
    #     truth_couple = truth_indices[truth_couple]
    #     pred_couple = pred_indices[pred_couple]
    #
    #     return truth_couple, pred_couple

    def compute_cost_matrix(self):
        """
        Not implemented.
        Needs to be implemented in super class.
        """
        raise NotImplementedError("Implement in supered class!")

    def get_coupled(self):
        """
        Retreives the coupled indices of the truth and predicted

        :returns : A `tuple` of truth and predicted coupled
        """
        return self.truth_couple, self.pred_couple

    def get_false_positives(self):
        """
        Retreives the indices of the false positive detections

        :returns : A `list` of indices that are false positive detections
        """
        if self.cost_matrix.shape[1] > 0:
            return numpy.array(list(set(range(self.cost_matrix.shape[1])) - set(self.pred_couple)))
        return numpy.array([])

    def get_false_negatives(self):
        """
        Retreives the indices of the false negative detections

        :returns : A `list` of indices that are false negative detections
        """
        if self.cost_matrix.shape[0] > 0:
            return numpy.array(list(set(range(self.cost_matrix.shape[0])) - set(self.truth_couple)))
        return numpy.array([])

    def get_score_summary(self, keys=None):
        """
        Computes all the scores in a `dict`

        :param keys: A `list` of scores to return

        :returns : A `dict` of scores
        """
        if isinstance(keys, (list, tuple)):
            return {
                key : getattr(self, key) for key in keys
            }
        return {
            "true_positive" : self.true_positive,
            "false_positive" : self.false_positive,
            "false_negative" : self.false_negative,
            "fnr" : self.fnr,
            "accuracy" : self.accuracy,
            "precision" : self.precision,
            "recall" : self.recall,
            "f1_score" : self.f1_score,
            "jaccard" : self.jaccard
        }

    @property
    def true_positive(self):
        """
        Computes the number of true positive

        :returns : The number of true positive
        """
        return len(self.pred_couple)

    @property
    def false_positive(self):
        """
        Computes the number of false_positive

        :returns : The number of false positive
        """
        return self.cost_matrix.shape[1] - self.true_positive

    @property
    def false_negative(self):
        """
        Computes the number of false negative

        :returns : The number of false negative
        """
        return self.cost_matrix.shape[0] - self.true_positive

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
        return 1 - self.recall

    @property
    def fpr(self):
        """
        Computes the false positive rate between the truth and the predictions
        NOTE. In the case of truth detections there are no true negatives

        :returns : A false positive rate
        """
        warnings.warn("Using the false positive rate as a score metric in the case truth predictions does not make sense as there are no true negative labels.",
                        category=UserWarning)
        return self.false_positive / self.false_negative

    @property
    def accuracy(self):
        """
        Computes the accuracy between the truth and the predictions

        :returns : An accuracy score
        """
        # No truth and no prediction is an accuracy of 1
        if all([shape == 0 for shape in self.cost_matrix.shape]):
            return 1.
        # Add numerical stability with 1e-6
        return len(self.pred_couple) / (self.cost_matrix.shape[0] + 1e-6)

    @property
    def precision(self):
        """
        Computes the precision between the truth and the predictions.

        :returns : A precision score
        """
        if all([shape == 0 for shape in self.cost_matrix.shape]):
            # Same behavior as in default sklearn
            return 0.
        # Add numerical stability with 1e-6
        return self.true_positive / (self.true_positive + self.false_positive + 1e-6)

    @property
    def recall(self):
        """
        Computes the recall between the truth and the predictions.

        :returns : A recall score
        """
        if all([shape == 0 for shape in self.cost_matrix.shape]):
            # Same behavior as in default sklearn
            return 0.
        # Add numerical stability with 1e-6
        return self.true_positive / (self.true_positive + self.false_negative + 1e-6)

    @property
    def f1_score(self):
        """
        Computes the F1-score between the truth and the predictions.

        :returns : A F1-score
        """
        prec = self.precision
        rec = self.recall
        return 2 * (prec * rec) / (prec + rec + 1e-6)

    @property
    def jaccard(self):
        """
        Computes the Jaccard Index between the truth and the predictions.

        :returns : A Jaccard index score
        """
        f1 = self.f1_score
        return f1 / (2 - f1)


class CentroidDetectionError(DetectionError):
    """
    Class implementing common detection scores based on the centroid of the
    segmented objects.
    """
    def __init__(self, truth, predicted, threshold, algorithm='nearest'):
        """
        Instantiates the `CentroidDetectionError` object

        :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
        :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
        :param threshold: A distance threshold to consider a true positive detection (pixels)
        :param algorithm: A `string` specifying the assignation algorithm to use, can be
                          either 'nearest' or 'hungarian'
        """
        super().__init__(
            truth=truth,
            predicted=predicted,
            algorithm=algorithm
        )

        self.truth_couple, self.pred_couple = self.assign(threshold=threshold)

    def compute_cost_matrix(self):
        """
        Computes the cost matrix between all objects
        """
        # Returns truth_couple and pred_couple to 0 if truth or predicted are empty
        if (len(self.truth) < 1) or (len(self.predicted) < 1):
            self.cost_matrix = numpy.ones((len(self.truth), len(self.predicted))) * 1e+6
        else:
            self.cost_matrix = spatial.distance.cdist(self.truth, self.predicted, metric='euclidean')

class IOUDetectionError(DetectionError):
    """
    Calculates the myriad of errors given a truth and prediction. The errors and
    default thresholds are taken from [caicedo2019].

    The idea behind this function is that you can calculate the detection errors given
    a minimal threshold of IOU. The errors you can obtain from this class are :
        1. The f1-score based on a minimal IOU (IOU >= 0.5)
        2. The type of errors that are made by the architecture
           (merged objects, split objects, missed objects or extra objects)

    @article{caicedo2019evaluation,
      title = {Evaluation of {{Deep Learning Strategies}} for {{Nucleus Segmentation}} in {{Fluorescence Images}}},
      author = {Caicedo, Juan C. and Roth, Jonathan and Goodman, Allen and Becker, Tim and Karhohs, Kyle W. and Broisin, Matthieu and Molnar, Csaba and McQuin, Claire and Singh, Shantanu and Theis, Fabian J. and Carpenter, Anne E.},
      year = {2019},
      volume = {95},
      pages = {952--965},
      issn = {1552-4930},
      doi = {10.1002/cyto.a.23863},
      annotation = {\_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/cyto.a.23863},
      copyright = {\textcopyright{} 2019 The Authors. Cytometry Part A published by Wiley Periodicals, Inc. on behalf of International Society for Advancement of Cytometry.},
      journal = {Cytometry Part A},
      language = {en},
      number = {9}
    }
    """
    def __init__(self, truth, predicted, algorithm="hungarian"):
        """
        Instantiates the `DetectionError` class

        :param truth: A `numpy.ndarray` of labeled objects
        :param predicted: A `numpy.ndarray` of labeled predicted objects
        """
        super().__init__(
            truth=truth,
            predicted=predicted,
            algorithm=algorithm
        )

        self.truth_couple, self.pred_couple = None, None

        self.compute_cost_matrix()

    def get_score_summary(self, threshold=0.5, keys=None):
        """
        Computes all the scores in a `dict`.

        NOTE. We need to recalculate the assignement to be ensure that truth_couple
              and pred_couple are well defined.

        :param threshold: (optional) A `float` of minimum threshold (default : 0.5)
        :param keys: (optional) A `list` of scores to return (default : all scores)

        :returns : A `dict` of scores
        """
        # Returns truth_couple and pred_couple to 0 if truth or predicted are empty
        if (not numpy.any(self.truth)) or (not numpy.any(self.predicted)):
            self.truth_couple, self.pred_couple = [], []
        else:
            self.truth_couple, self.pred_couple = self.assign(threshold=threshold, maximize=True)

        if isinstance(keys, (list, tuple)):
            return {
                key : getattr(self, key) for key in keys
            }
        return {
            "true_positive" : self.true_positive,
            "false_positive" : self.false_positive,
            "false_negative" : self.false_negative,
            "fnr" : self.fnr,
            "accuracy" : self.accuracy,
            "precision" : self.precision,
            "recall" : self.recall,
            "f1_score" : self.f1_score,
            "jaccard" : self.jaccard
        }

    def get_f1_score(self, threshold=0.5):
        """
        Computes the F1-score from the cost matrix given a minimum threshold.
        The default threshold is 0.5 since it is commonly used.

        :param threshold: A `float` of the minimum threshold
                          OR A `list` of the minimum thresholds

        :returns : A `numpy.ndarray` of the F1-scores
                   A `numpy.ndarray` of the thresholds
        """
        if isinstance(threshold, float):
            thresholds = [threshold]
        elif isinstance(threshold, (list, tuple, numpy.ndarray)):
            thresholds = threshold

        f1_scores = []
        for threshold in thresholds:

            # Returns truth_couple and pred_couple to 0 if truth or predicted are empty
            if (not numpy.any(self.truth)) or (not numpy.any(self.predicted)):
                self.truth_couple, self.pred_couple = [], []
            else:
                self.truth_couple, self.pred_couple = self.assign(threshold=threshold, maximize=True)

            f1_scores.append(self.f1_score)
            # cost_matrix = self.cost_matrix >= threshold
            #
            # # A truth is associated to a prediction
            # true_positive = numpy.sum(cost_matrix, axis=1) >= 1
            # true_positive = true_positive.sum()
            #
            # # A truth is not associated to any prediction
            # false_negative = numpy.sum(cost_matrix, axis=1) == 0
            # false_negative = false_negative.sum()
            #
            # # A prediction is not associated to any truth
            # false_positive = numpy.sum(cost_matrix, axis=0) == 0
            # false_positive = false_positive.sum()
            #
            # f1 = 2 * true_positive / (2 * true_positive + false_negative + false_positive)
            # f1_scores.append(f1)

        return numpy.array(f1_scores), numpy.array(thresholds)

    def get_missed_objects(self, threshold=0.7):
        """
        Gets the missed objects and returns a list of objects
        regionprops.
        This corresponds to false negative detections.

        :param threshold: A `float` to threshold the cost matrix

        :returns : A `list` of regionprops of the missed objects
        """
        cost_matrix = self.cost_matrix >= threshold
        regionprops = measure.regionprops(self.truth)
        false_negatives = self.get_false_negatives()
        return [regionprops[fn] for fn in false_negatives]

    def get_extra_objects(self, threshold=0.7):
        """
        Gets the extra objects induced and returns a list of objects
        regionprops.
        This corresponds to false positive detections.

        :param threshold: A `float` to threshold the cost matrix

        :returns : A `list` of regionprops of the extra objects
        """
        cost_matrix = self.cost_matrix >= threshold
        regionprops = measure.regionprops(self.predicted)
        false_positives = self.get_false_positives()
        return [regionprops[fp] for fp in false_positives]

    def get_split_objects(self, threshold=0.1):
        """
        Gets the split objects and returns a list of object regionprops.
        This corresponds to truth masks that cover multiple predicted objects
        with a minimal threshold.

        :param threshold: A `float` to threshold the cost matrix

        :returns : A `list` of regionprops of the split truth objects
        """
        cost_matrix = self.cost_matrix >= threshold
        regionprops = measure.regionprops(self.truth)
        split_objects = numpy.sum(cost_matrix, axis=1) >= 2 # at least 2 detections
        return [rprop for rprop, so in zip(regionprops, split_objects) if so]

    def get_merged_objects(self, threshold=0.1):
        """
        Gets the merged objects.
        This corresponds to predicted masks that cover multiple objects
        with a minimal threshold.

        :param threshold: A `float` to threshold the cost matrix

        :returns : A `list` of regionprops of the merged truth objects
        """
        cost_matrix = self.cost_matrix >= threshold
        regionprops = measure.regionprops(self.predicted)
        merged_objects = numpy.sum(cost_matrix, axis=0) >= 2 # at least 2 detections
        return [rprop for rprop, mo in zip(regionprops, merged_objects) if mo]

    def compute_cost_matrix(self):
        """
        Computes the cost matrix between all objects
        """
        # Masks without 0
        masks_truth = self.truth == numpy.unique(self.truth)[1:, numpy.newaxis, numpy.newaxis]
        masks_prediction = self.predicted == numpy.unique(self.predicted)[1:, numpy.newaxis, numpy.newaxis]

        self.cost_matrix = numpy.zeros((len(masks_truth), len(masks_prediction)))
        for j, mask_truth in enumerate(masks_truth):
            for i, mask_prediction in enumerate(masks_prediction):
                if numpy.any(mask_truth * mask_prediction):
                    iou = commons.iou(mask_truth, mask_prediction)
                    self.cost_matrix[j, i] = iou
                else:
                    self.cost_matrix[j, i] = 0.

    def show(self, threshold=0.5, axes=None):
        """
        Implements a show function of the truth and predicted

        :param threshold: (optional) A `float` of the threshold to use (default : 0.5)
        :param ax: (optional) A `tuple` of size (2, ) of `matplotlib.Axes`
        """
        # Returns truth_couple and pred_couple to 0 if truth or predicted are empty
        if (not numpy.any(self.truth)) or (not numpy.any(self.predicted)):
            self.truth_couple, self.pred_couple = [], []
        else:
            self.truth_couple, self.pred_couple = self.assign(threshold=threshold, maximize=True)

        missed_objs = self.get_missed_objects()
        extra_objs = self.get_extra_objects()
        split_objs = self.get_split_objects()
        merged_objs = self.get_merged_objects()

        # Creates the axes
        if isinstance(axes, type(None)):
            fig, axes = pyplot.subplots(1, 2)

        axes[0].imshow(self.truth)
        for obj in missed_objs:
            min_row, min_col, max_row, max_col = obj.bbox
            rect = patches.Rectangle(
                xy = (min_col, min_row),
                width = max_col - min_col,
                height = max_row - min_row,
                facecolor="none", edgecolor="red"
            )
            axes[0].add_artist(rect)
        for obj in split_objs:
            min_row, min_col, max_row, max_col = obj.bbox
            rect = patches.Rectangle(
                xy = (min_col, min_row),
                width = max_col - min_col,
                height = max_row - min_row,
                facecolor="none", edgecolor="yellow"
            )
            axes[0].add_artist(rect)

        axes[1].imshow(self.predicted)
        for obj in extra_objs:
            min_row, min_col, max_row, max_col = obj.bbox
            rect = patches.Rectangle(
                xy = (min_col, min_row),
                width = max_col - min_col,
                height = max_row - min_row,
                facecolor="none", edgecolor="red"
            )
            axes[1].add_artist(rect)
        for obj in merged_objs:
            min_row, min_col, max_row, max_col = obj.bbox
            rect = patches.Rectangle(
                xy = (min_col, min_row),
                width = max_col - min_col,
                height = max_row - min_row,
                facecolor="none", edgecolor="yellow"
            )
            axes[1].add_artist(rect)

        axes[0].legend(
            handles=[patches.Patch(facecolor="red"), patches.Patch(facecolor="yellow")],
            labels=["Missed objects", "Split objects"],
            loc="lower left"
        )
        axes[1].legend(
            handles=[patches.Patch(facecolor="red"), patches.Patch(facecolor="yellow")],
            labels=["Extra objects", "Merged objects"],
            loc="lower left"
        )

if __name__ == '__main__':

    #############################################
    # Test CentroidDetectionError
    #############################################

    dmatch = 5
    nb_pred = 90
    nb_gt = 100

    numpy.random.seed(2)

    truth = numpy.random.random_sample((nb_gt, 2)) * 100
    pred = numpy.random.random_sample((nb_pred, 2)) * 100

    s = CentroidDetectionError(truth, pred, dmatch, algorithm='nearest')
    print(f'Truth index {s.truth_couple}')
    print(f'Pred index {s.pred_couple}')
    # print(f'False negatives {s.get_false_negatives()}')
    # print(f'False positives {s.get_false_positives()}')

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
        # print(f'GT {s.truth_couple[i]}, pred {s.pred_couple[i]}, dist {numpy.linalg.norm(x1 - x2)}')

    s = CentroidDetectionError(truth, pred, dmatch, algorithm='hungarian')
    print(f'Truth index {s.truth_couple}')
    print(f'Pred index {s.pred_couple}')
    print(s.get_score_summary())
    # print(f'False negatives {s.get_false_negatives()}')
    # print(f'False positives {s.get_false_positives()}')

    ax = axes[1]
    ax.scatter(truth[:, 0], truth[:, 1], marker='o')
    ax.scatter(pred[:, 0], pred[:, 1], marker='x')
    for i in range(len(s.truth_couple)):
        x1 = truth[s.truth_couple[i]]
        x2 = pred[s.pred_couple[i]]
        ax.text(*x1, str(s.truth_couple[i]))
        ax.text(*x2, str(s.pred_couple[i]))
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], 'r-')
        # print(f'GT {s.truth_couple[i]}, pred {s.pred_couple[i]}, dist {numpy.linalg.norm(x1 - x2)}')
    plt.show()

    #############################################
    # Test IOUDetectionError
    #############################################

    from skimage import draw
    from matplotlib import pyplot, patches

    truth, _ = draw.random_shapes(
        image_shape=(256, 256),
        max_shapes=10,
        min_shapes=5,
        min_size=25,
        multichannel=False,
        random_seed=42
    )
    truth = abs(255 - truth)

    predicted, _ = draw.random_shapes(
        image_shape=(256, 256),
        max_shapes=15,
        min_shapes=5,
        min_size=25,
        multichannel=False,
        random_seed=42
    )

    # predicted = abs(255 - predicted)
    # predicted = numpy.roll(truth, 5, axis=1)
    predicted = truth

    # Creates a split obj
    predicted[:, 140] = 0
    predicted = measure.label(predicted)

    # Creates a merged obj
    truth[:, 35] = 0
    truth = measure.label(truth)

    s = IOUDetectionError(truth, predicted, algorithm="hungarian")
    f1_scores, threshold = s.get_f1_score(threshold=[0.5])
    print(s.get_score_summary())

    s.show()

    pyplot.show()
