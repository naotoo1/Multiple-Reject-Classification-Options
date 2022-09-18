[![Python: 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch: 1.11](https://img.shields.io/badge/pytorch-1.11-orange.svg)](https://pytorch.org/blog/pytorch-1.11-released/)
[![Prototorch: 0.7.3](https://img.shields.io/badge/prototorch-0.7.3-blue.svg)](https://pypi.org/project/prototorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# Multiple-Reject-Classification-Options
Prototype and non prototype-based ML implementation for determining class related thresholds used in multiple reject classification strategy for improving classification reliability in scientific technical or high risk areas of ML models utilization.

## What is it?
In Machine learning, the ubiquitous convention has been to train a model to give accurate/reliable predictions on all available test data. The reality of this convention is far from the truth since trained models may sometimes give unexpected/unwanted results. 

In areas requiring high reliability, ML models should be able to say no idea to low confident decisions. Technically, ML models must learn to reject low confidence predictions when there is evidence of uncertainty about some decisions made. 

This sincerity reduces the risk by increasing the reliability of ML models, opening the door for further investigation and new strategies to predict the rejected test data points.

## How to use
The implementation of the constrained optimization problem where users want a very low classification rejection rate and high model performance is shown ```crt.py```
An example can be found in ```crt_chow_bcd.py``` where a simulation has been done to test the performance to the implemented algorithm for the chow method and the crt method.

An advance example on the application of class related thresh-holds in the ensemble diagnosis of breast cancer disease is shown below:

Import some libraries
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from prosemble import Hybrid, ThreshT, ProtoCert, ProtoCertt, visualize
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

load some ```train models``` as pickle files
```python
pickle_in1 = open("svc.pkl", "rb")
pickle_in2 = open("knn.pkl", "rb")
pickle_in3 = open("dtc.pkl", "rb")

svc = pickle.load(pickle_in1)
knn = pickle.load(pickle_in2)
dtc = pickle.load(pickle_in3)
```

Get access to the test data set
```python
scaler = StandardScaler()
X, y = load_breast_cancer(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape)
```

Define a function to return the respective confidence of the train models competing in the ensemble
```python
def get_posterior(x, y_, z_):
    """

    :param x: Input data
    :param y_: prediction
    :param z_: model
    :return: prediction probabilities
    """
    z1 = z_.predict_proba(x)
    certainties = [np.max(i) for i in z1]
    cert = np.array(certainties).flatten()
    cert = cert.reshape(len(cert), 1)
    y_ = y_.reshape(len(y_), 1)
    labels_with_certainty = np.concatenate((y_, cert), axis=1)
    return np.round(labels_with_certainty, 4)
```

Define a function to return the confidence of the ensemble model
```python
def get_ens_confidence(x, y):
    """

    :param x: predicted labels from the ensemble
    :param y: predicted confidence from the ensemble
    :return: [x,y]
    """
    x = np.array(x)
    x = x.reshape(len(x), 1)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    ens_confidence = np.concatenate((x, y), axis=1)
    return ens_confidence
```

Define a function to return rejected test data points
```python
def get_rejected_data(x, y):
    r_data = []
    for i in x:
        r_data.append(y[i])
    return np.array(r_data)
```

Set up the ensemble model using the Hybrid class from the prosemble ML tool box
```python
# classes labels
proto_classes = np.array([0, 1])

# object of Hybrid class from prosemble
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')
```

Get predictions and the confidence from the respective models in the ensemble
```python
# predictions using the svc,knn and dtc models respectively
pred1 = svc.predict(X_test)
pred2 = knn.predict(X_test)
pred3 = dtc.predict(X_test)

# confidence of prediction using the svc,knn and dtc models respectively
sec1 = get_posterior(x=X_test, y_=pred1, z_=svc)
sec2 = get_posterior(x=X_test, y_=pred2, z_=knn)
sec3 = get_posterior(x=X_test, y_=pred3, z_=dtc)
all_pred = [pred1, pred2, pred3]
all_sec = [sec1, sec2, sec3]
```

Get the predictions and confidence from the ensemble model using either the hard or soft voting method. In this example we would consider the soft method
```python
# predicted labels from the ensemble using hard voting
prediction1 = ensemble.pred_prob(X_test, all_pred)
# predicted labels from the ensemble using soft voting
prediction2 = ensemble.pred_sprob(X_test, all_sec)
# confidence of the prediction using hard voting
hard_prob = ensemble.prob(X_test, all_pred)
# confidence of the prediction using soft voting
soft_prob = ensemble.sprob(X_test, all_sec)
```
#### CRT Approach 
Set-up the object instance for class related confidence thresh-hold using the ThreshT class from prosemble ML package. NB: The rejection rate determines the maximum percentage of the test cases for which the ensemble model may reject based on low confidence decisions.
```python
crt = ThreshT(predict_results=prediction1, reject_rate1=0.1)
```

Set-up the object instance for sorting based on crt algorithm using the ProtoCert class from prosemble ML package
```python
protocert = ProtoCert(y_test=y_test)
```

Get the confidence thresh-hold for each class using the crt algorithm
```python
# predicted labels from the ensemble along with the confidence.
d1 = get_ens_confidence(prediction2, soft_prob)

# class related confidence thresh-hold for Malignant
h1 = crt.thresh_new(d1=d1, protocert_1=protocert, j=0)
#  class related confidence thresh-hold for Benign
h2 = crt.thresh_new(d1=d1, protocert_1=protocert, j=1)
```

Set-up the object instance needed to do sorting for final results using the ProtoCertt class from prosemble.
```python
protocertt = ProtoCertt(y_test=y_test)
```

Determine non-rejected classifications based on the class realted thresh-holds determined by the crt algorithm and check performance as against when the ensemble model predicted on all the test cases.
```python
non_rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='>=', y__='l', l3=[0, 1])
index_non_rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='>=', y__='i', l3=[0, 1])
true_labelsN = protocertt.thresh_y_test(x=index_non_rejected_labels)

# accuracy of model without rejection
accuracy1 = accuracy_score(y_true=y_test, y_pred=prediction2)

# accuracy of model with rejection max accepted rejection rate of 10% of test cases w.r.t crt
accuracy = accuracy_score(y_true=true_labelsN, y_pred=non_rejected_labels)
```
Get access to the rejected test cases for further investigation or for further considerations in deciding on the diagnosis.
```python
# Access rejected emsemble predicted labels
rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='<', y__='l', l3=[0, 1])

# Access index of rejected ensemble labels
index_rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='<', y__='i', l3=[0, 1])

# Access rejected data points in the test data.
print(get_rejected_data(x=index_rejected_labels, y=X_test))
```

#### Chow's Approach 
choose a universal confidence thresh-hold for all the classes based on a prior knowledge(chow)

Set-up the object instance for class responsible for sorting based on chow's approach
```python
protocert = ProtoCert(y_test=y_test)
```
Get predicted labels from the ensemble along with the confidence.
```python
d1 = get_ens_confidence(prediction2, soft_prob)
```
Determine non-rejected classifications based on the class realted thresh-holds determined by the chow's approach and check performance as against when the ensemble model predicted on all the test cases.
one may fall on the visualization tool in prosemble to determine to universal confidence thresh-hold

```python
post_confidences_0 = protocert.thresh_function(x=d1, y=0, y_='>=', y__='s', y___=0)
post_confidences_1 = protocert.thresh_function(x=d1, y=0, y_='>=', y__='s', y___=1)
posterior_confidence = [post_confidences0, post_confidences1]
```
Visualize the posterior label securities 
```python
vis = visualize(confidence_list=posterior_confidence, num_classes=2, colors=['#00FF00', '#FF00FF'],
                class_labels=['class 0', 'class 1'])

# get summary visualization for all classes
vis.get_vis(x='plot for posterior label securities')

# get visualization for each class
vis.get_vis_(x='Confidence', y='Frequency', z='Evaluation of confidence threshold for')

```

```python
# choose a universal confidence thresh-hold for all the classes based on a prior knowledge(chow)
non_rejected_labels = protocert.thresh_function(x=d1, y=0.8, y_='>=', y__='l', y___=None)
index_non_rejected_labels = protocert.thresh_function(x=d1, y=0.8, y_='>=', y__='i', y___=None)
true_labelsN = protocert.thresh_y_test(x=index_non_rejected_labels)

# accuracy of model without rejection
accuracy1 = accuracy_score(y_true=y_test, y_pred=prediction2)

# accuracy of model with rejection based on chows appraoch
accuracy = accuracy_score(y_true=true_labelsN, y_pred=non_rejected_labels)
```
Get access to the rejected test cases for further investigation or for further considerations in deciding on the diagnosis.

```python
# Access rejected emsemble predicted labels
rejected_labels = protocert.thresh_function(x=d1, y=0.8, y_='<', y__='l', y___=None)

# Access index of rejected labels
index_rejected_labels = protocert.thresh_function(x=d1, y=0.8, y_='<', y__='i', y___=None)

# Access rejected data points in the test data.
print(get_rejected_data(x=index_rejected_labels, y=X_test))
```

## Simulation

A simulated result from multiple reject thresholds for improving classification reliability using the CRT vs Chow is shown in the figure below for GLVQ using the breast cancer diagnostic data.

Even though the chow method is known to produce optimal results, its shortcomings exist in the fact that users will most definitely not have access to the prior knowledge of the confidence thresh-holds to be used universally for all classes in the test cases. The significance of the CRT approach lies in the option which allows users to provide a prior maximum rejection rate that is readily available to the users.

We observe below that the CRT method efficiently models chows performance and even beat it in the long run. Hence the CRT approach provides a better option in the classification reject strategy. This has been demonstrated with the implementation of CRT as part of the [prosemble ML package](https://github.com/naotoo1/Multiple-Reject-Classification-Options) and tested in practical advance use case for the  breast cancer diagnosis study.


![Figure_1](https://user-images.githubusercontent.com/82911284/173432371-74790b50-f264-46c6-aecd-49b7700ace4a.png)

## References

<a id="1">[1]</a> 
Fumera, G., Roli, F., & Giacinto, G. (2000, August). 
Multiple reject thresholds for improving classification reliability. 
In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 863-871). Springer, Berlin, Heidelberg.

