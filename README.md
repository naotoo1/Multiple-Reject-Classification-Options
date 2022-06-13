[![Python: 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch: 1.11](https://img.shields.io/badge/pytorch-1.11-orange.svg)](https://pytorch.org/blog/pytorch-1.11-released/)
[![Prototorch: 0.7.3](https://img.shields.io/badge/prototorch-0.7.3-blue.svg)](https://pypi.org/project/prototorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# Multiple-Reject-Classification-Options
Prototype and non prototype-based ML implementation for determining class related thresholds used in multiple reject classifiaction strategy for improving classification reliability in scientific technical or high risk areas of ML models utilization

## What is it?
In Machine learning, the ubiquitous convention has been to train a model to give accurate/reliable predictions on all available test data. The reality of this convention is far from the truth since trained models may sometimes give unexpected/unwanted results. 

In areas requiring high reliability, ML models should be able to say no idea to low confident decisions. Technically, ML models must learn to reject low confidence predictions when there is evidence of uncertainty about some decisions made. 

This sincerity reduces the risk by increasing the reliability of ML models, opening the door for further investigation and new strategies to predict the rejected test data points.

## How to use
The implementation of the constrained optimization problem where users want a very low classification rejection rate and high model performance is shown ```crt.py```
An example can be found in ```crt_chow_bcd.py``` where a simulation has been done to test the performance to the implemented algorithm for the chow method and the crt method.


## Simulation

A simulated results from multiple reject thresholds for improving classification reliability using the CRT vs Chow is shown below for GLVQ using the breast cancer diagnostic data

![Figure_1](https://user-images.githubusercontent.com/82911284/173432371-74790b50-f264-46c6-aecd-49b7700ace4a.png)

## References

<a id="1">[1]</a> 
Fumera, G., Roli, F., & Giacinto, G. (2000, August). 
Multiple reject thresholds for improving classification reliability. 
In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 863-871). Springer, Berlin, Heidelberg.

