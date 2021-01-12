
import numpy

from scipy import stats

def correlation_coeff(truth, prediction, mask=None, normality=True):
    """
    Calculates the pearson correlation coefficients between predicted intensity
    and the ground truth intensity. The correlation coefficient is bounded in range
    [-1, 1]. We aim at obtaining a high correlation coefficient between the truth
    and the prediction.

    :param truth: A `numpy.ndarray` of the ground truth pixel intensity
    :param prediction: A `numpy.ndarray` of the predicted pixel intensity
    :param mask: (Optional) A `numpy.ndarray` of specific regions where to calculate
                            the pearson coefficient
    :param normality: (Optional) Whether we should consider normality to calculate
                                 the correlation coefficients

    :returns : The correlation coefficient between pixel intensities
               The p-value for hypothesis test that samples are uncorrelated
    """
    if isinstance(mask, type(None)):
        mask = numpy.ones_like(truth, dtype=bool)
    else:
        mask = mask.astype(bool)

    x, y = truth[mask], prediction[mask]
    if normality:
        r, pvalue = stats.pearsonr(x, y)
    else:
        r, pvalue = stats.spearmanr(x, y)
    return r, pvalue

if __name__ == "__main__":

    truth = numpy.random.rand(256, 256)
    prediction = numpy.random.rand(256, 256)
    mask = numpy.zeros_like(truth, dtype=bool)
    mask[:100, :100] = True

    print(correlation_coeff(truth, prediction, mask=mask, normality=True))
    print(correlation_coeff(truth, prediction, mask=mask, normality=False))

    print(correlation_coeff(truth, truth, normality=True

    prediction = truth + 10
    print(correlation_coeff(truth, prediction, mask=mask, normality=True))
