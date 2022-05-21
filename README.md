[![Python: 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch: 1.11](https://img.shields.io/badge/pytorch-1.11-orange.svg)](https://pytorch.org/blog/pytorch-1.11-released/)
[![Prototorch: 0.7.3](https://img.shields.io/badge/prototorch-0.7.3-blue.svg)](https://pypi.org/project/prototorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# Multiple-Reject-Classification-Options
Prototype based ML implementation for determining class related thresholds used in multiple reject classifiaction Strategy

## How to use
The implementation of the constrained optimization problem where users want a very low classification rejection rate and high model performance is shown ```crt.py```
An example can be found in ```crt_chow_iris.py```


## Simulation

A simulated results from multiple reject thresholds for improving classification reliability using the CRT vs Chow is shown below for GLVQ

![Figure_4](https://user-images.githubusercontent.com/82911284/165868376-edfd75fd-dc67-41f1-a75a-305f2a72e06f.png)


## References

<a id="1">[1]</a> 
Fumera, G., Roli, F., & Giacinto, G. (2000, August). 
Multiple reject thresholds for improving classification reliability. 
In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 863-871). Springer, Berlin, Heidelberg.

