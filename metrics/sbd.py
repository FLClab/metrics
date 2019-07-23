
import numpy

from skimage import draw, morphology
from scipy import signal
from matplotlib import pyplot

def boundary(mask, radius):
    """
    Computes the boundary of the mask with given radius

    :param mask: A 2D binary `numpy.ndarray`
    :param radius: A radius in which to look for neighborhood pixels

    :returns : A 2D binary `numpy.ndarray` of the boundary
    """
    radius = radius // 2
    disk = morphology.disk(radius)
    out = signal.convolve2d(mask, morphology.disk(radius), mode="same")
    return (out != 0) & (out != disk.sum())

def DSC(truth, predicted):
    """
    Computes the dice similarity coefficient

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction

    :returns : Dice similarity coefficient
    """
    truth, predicted = truth.astype(bool), predicted.astype(bool)
    intersection = (truth * predicted).sum()
    # print(2 * intersection / (truth.sum() + predicted.sum()), intersection, truth.sum(), predicted.sum())
    return 2 * intersection / (truth.sum() + predicted.sum())

def SBD(truth, predicted, radius=2, foreground=None, **kwargs):
    """
    Computes the symmetric boundary dice of the truth and predicted [1]

    :param truth: A 2D binary `numpy.ndarray` of ground truth
    :param predicted: A 2D binary `numpy.ndarray` of prediction
    :param radius: A radius in which to look for neighborhood pixels

    :returns : Symmetric boundary dice score

    [1]. Varduhi Yeghiazaryan, Irina Voiculescu, “Family of boundary overlap metrics for the evaluation of medical
         image segmentation,” J. Med. Imag. 5(1), 015006 (2018), doi: 10.1117/1.JMI.5.1.015006.
    """
    def subarray(mask, pos, radius):
        """
        Slices an array at a given center position with a given radius

        :param mask: A 2D binary `numpy.ndarray`
        :param pos: A `tuple` of position
        :param radius: Half the size of the sliced array
        """
        return mask[max(0, pos[0] - radius) : min(mask.shape[0], pos[0] + radius),
                    max(0, pos[1] - radius) : min(mask.shape[1], pos[1] + radius)]
    if isinstance(foreground, (type(None))):
        foreground = numpy.ones(truth.shape)
    truth, predicted = (truth * foreground).astype(bool), (predicted * foreground).astype(bool)
    bound_truth, bound_pred = boundary(truth, radius=radius), boundary(predicted, radius=radius)
    dice = []
    where_true, where_pred = numpy.argwhere(bound_truth), numpy.argwhere(bound_pred)
    for pos in numpy.vstack((where_true, where_pred)):
        dice.append(DSC(subarray(truth, pos=pos, radius=radius),
                        subarray(predicted, pos=pos, radius=radius)))
    return numpy.mean(dice)

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
        (325, 300),
        (500, 320),
        (420, 430),
        (250, 590),
        (350, 300),
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

    score = SBD(image1, image2, radius=10)
    print(score)

    image3 = numpy.ones((28, 28))
    bound = boundary(image3, 3)
    fig, ax = pyplot.subplots()
    ax.imshow(bound.astype(int))
    pyplot.show()