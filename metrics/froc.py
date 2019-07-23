
import numpy

from matplotlib import pyplot
from skimage import draw, filters, measure, morphology
from scipy import spatial, optimize

def compute_fr(truth_labels, pred_labels, dist_thres):

    truth_props = measure.regionprops(truth_labels)
    pred_props = measure.regionprops(pred_labels)

    distances = spatial.distance.cdist([prop.centroid for prop in pred_props],
                                       [prop.centroid for prop in truth_props])
    if distances.shape[1] == 1:
        assignment = distances.ravel()
    else:
        row_ind, col_ind = optimize.linear_sum_assignment(distances)
        assignment = distances[row_ind, col_ind]
    assignment = assignment < numpy.repeat(dist_thres, assignment.size)

    return assignment.sum(), len(pred_props) - assignment.sum(), len(truth_props)

def FROC(truth, predicted, dist_thres=3):

    truth_labels = measure.label(truth.astype(int))
    predicted_max = filters.rank.maximum((predicted * 255).astype(numpy.uint8), selem=morphology.square(5))
    predicted = predicted.copy()
    predicted[predicted < 0.99 * (predicted_max / 255.)] = 0

    thresholds = numpy.linspace(0.1, 0.9, num=25)
    tps, fps, number_positives = [], [], []
    for threshold in thresholds:
        pred_labels = measure.label((predicted >= threshold).astype(int))
        tp, fp, number_positive = compute_fr(truth_labels, pred_labels, dist_thres)
        tps.append(tp), fps.append(fp), number_positives.append(number_positive)
    return tps, fps, number_positives

if __name__ == "__main__":

    from tqdm import tqdm
    import random

    truths, predictions = [], []
    for _ in range(30):
        centers = numpy.random.randint(12, 500, size=(random.randint(20, 64), 2))
        truth = numpy.zeros((512, 512))
        for (r, c) in centers[:random.randint(len(centers)-1, len(centers) + 1)]:
            rr, cc = draw.circle(r, c, radius=5, shape=truth.shape)
            truth[rr, cc] = 1

        predicted = numpy.zeros((512, 512))
        predicted[centers[:, 0], centers[:, 1]] = 1
        predicted = filters.gaussian(predicted, sigma=5)
        predicted /= predicted.max()

        truths.append(truth), predictions.append(predicted)

    TP, FP, nums = [], [], []
    for truth, predicted in tqdm(zip(truths, predictions)):
        tp, fp, num = FROC(truth, predicted, 3)
        TP.append(tp), FP.append(fp), nums.append(num)

    fig, ax = pyplot.subplots()
    ax.plot([*(numpy.sum(FP, axis=0) / len(truths)), 0],
            [*(numpy.sum(TP, axis=0) / numpy.sum(nums, axis=0)), 0], marker="o")

    ax.set_ylim(0, 1)
    pyplot.show()
