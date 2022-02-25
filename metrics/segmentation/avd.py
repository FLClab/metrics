
import numpy

from skimage import draw, morphology
from scipy import signal, spatial
from matplotlib import pyplot

from metrics import utils

def remove_intersection(truth, predicted):
    """
    Removes the intersection between truth and predictions from truth

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :returns : A 2D binary `numpy.ndarray` of truth without the intersection
    """
    intersection = truth * predicted
    return truth * ~intersection

def directed_avg_Hausdorff(truth, predicted):
    """
    Computes the directed average Hausdorff distance between truth and predicted

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :return : The directed average Hausdorff distance
    """
    nointer_truth = remove_intersection(truth, predicted)
    bound_pred = utils.boundary(predicted)
    if (not numpy.any(bound_pred)) or (not numpy.any(truth)):
        return 0
    where_true, where_pred = numpy.argwhere(nointer_truth), numpy.argwhere(bound_pred)
    distances = spatial.distance.cdist(where_true, where_pred)
    return numpy.sum(numpy.min(distances, axis=1)) / truth.sum()

def AVD(truth, predicted):
    """
    Computes the average Hausdorff Distance (or average distance) [1]

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :returns : The average Hausdorff distance

    [1]. Taha, A.A. and Hanbury, A., 2015. Metrics for evaluating 3D medical image
         segmentation: analysis, selection, and tool. BMC medical imaging, 15(1), p.29.
    """
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    return max(directed_avg_Hausdorff(truth, predicted), directed_avg_Hausdorff(predicted, truth))

if __name__ == "__main__":

    image1 = numpy.zeros((512, 512))
    poly = numpy.array((
        (300, 300),
        (480, 320),
        (380, 430),
        (220, 590),
        (300, 300),
    ))
    rr, cc = draw.polygon(poly[:, 0], poly[:, 1], image1.shape)
    image1[rr, cc] = 1
    rr, cc = draw.circle(200, 200, 100, image1.shape)
    image1[rr, cc] = 1
    rr, cc = draw.circle(200, 200, 25, image1.shape)
    image1[rr, cc] = 0

    image2 = numpy.zeros((512, 512))
    poly = numpy.array((
        (310, 300),
        (490, 320),
        (390, 430),
        (230, 590),
        (310, 300),
    ))
    rr, cc = draw.polygon(poly[:, 0], poly[:, 1], image2.shape)
    image2[rr, cc] = 1
    rr, cc = draw.circle(225, 200, 100, image2.shape)
    image2[rr, cc] = 1
    rr, cc = draw.circle(225, 225, 25, image2.shape)
    image2[rr, cc] = 0

    fig, ax = pyplot.subplots(1,2)
    ax[0].imshow(image1)
    ax[1].imshow(image2)

    score = AVD(image1, image2)
    print(score)
    pyplot.show()
