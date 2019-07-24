
from scipy import signal
from skimage import morphology

def boundary(mask, radius=1):
    """
    Computes the boundary of the mask with given radius

    :param mask: A 2D binary `numpy.ndarray`
    :param radius: A radius in which to look for neighborhood pixels

    :returns : A 2D binary `numpy.ndarray` of the boundary
    """
    selem = morphology.square(radius * 2)
    out = signal.convolve2d(mask, selem, mode="same")
    return (out != 0) & (out != selem.sum())

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
