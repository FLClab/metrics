
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
