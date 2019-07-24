# metrics

This package contains simple implementations of common metrics used in segmentation. The implemented metrics all require binary ground truths and predictions. Depending on the metric, it can also require additionnal parameters or predictions in range \[0, 1\].

## Install

To install the package, _i.e._ to be able to use localy this package, one should do the following 
```bash
git clone https://github.com/antho214/metrics
pip install -e metrics
```
To verify the installation one should try to import pygithub from a python console.
```python
import metrics
```

## HOW-TO

This section briefly describes how to use the package 
```python
import metrics
metrics.SBD(truth, prediction)
```
where `truth` and `prediction` are of type `numpy.ndarray`.

## Collaborate

To implement a metric in this package, a user should create a python module in folder `./metrics/` and associate this module in the `__init__` file of the folder. As a mean of consistency of the package, a module should be lowercased while a function should be uppercased. For example, if a user wanted to implement a mean square error (MSE) module
```python
% inside mse.py 
import numpy 
def MSE(truth, prediction):
    """
    Computes the mean square error between ground truth and prediction
    
    :param truth: A 2D binary `numpy.ndarray` of ground truth 
    :param prediction: A 2D `numpy.ndarray` of prediction 
    
    :returns : The mean squared error between ground truth and prediction
    """
    return numpy.mean((truth - prediction)**2)
```
and inside the `__init__` file the user should add the following 
```python 
from .mse import MSE
```
