
import numpy

from matplotlib import pyplot
from skimage import draw, filters, measure, morphology
from scipy import spatial, optimize

def compute_fr(truth, predicted, threshold, dist_thres):
    truth_labels = measure.label(truth)
    predicted_max = filters.rank.maximum((predicted * 255).astype(numpy.uint8), selem=morphology.square(3))
    temp = predicted.copy()
    temp[predicted < 0.99 * (predicted_max / 255.)] = 0
    pred_labels = measure.label((temp >= threshold).astype(int))

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

    return assignment.sum() / len(truth_props), len(pred_props) - assignment.sum()

def FROC(truth, predicted, dist_thres=3):
    thresholds = numpy.linspace(0.1, 0.9, num=25)
    TPR, FP = [], []
    for threshold in thresholds:
    # for threshold in [0.5]:
        tpr, fp = compute_fr(truth, predicted, threshold, dist_thres)
        TPR.append(tpr), FP.append(fp)
    return TPR, FP

if __name__ == "__main__":

    from tqdm import tqdm
    import random

    truths, predictions = [], []
    for _ in range(5):
        centers = numpy.random.randint(12, 500, size=(25, 2))
        truth = numpy.zeros((512, 512))
        for (r, c) in centers[:-random.randint(0,2)]:
            rr, cc = draw.circle(r, c, radius=5, shape=truth.shape)
            truth[rr, cc] = 1

        predicted = numpy.zeros((512, 512))
        i = random.randint(0,2)
        predicted[centers[i:, 0] + 1, centers[i:, 1] + 2] = 1
        predicted = filters.gaussian(predicted, sigma=5)
        predicted /= predicted.max()

        truths.append(truth), predictions.append(predicted)

    TPR, FP = [], []
    for truth, predicted in tqdm(zip(truths, predictions)):
        tpr, fp = FROC(truth, predicted, 3)
        TPR.append(tpr), FP.append(fp)

    fig, ax = pyplot.subplots()
    ax.plot([*numpy.mean(FP, axis=0), 0], [*numpy.mean(TPR, axis=0), 0])

    pyplot.show()
