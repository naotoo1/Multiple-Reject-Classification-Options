# import libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from prosemble import Hybrid, ThreshT, ProtoCert, ProtoCertt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load train models as pickle files
pickle_in1 = open("svc.pkl", "rb")
pickle_in2 = open("knn.pkl", "rb")
pickle_in3 = open("dtc.pkl", "rb")

svc = pickle.load(pickle_in1)
knn = pickle.load(pickle_in2)
dtc = pickle.load(pickle_in3)

#  summary of models in the ensemble
models = [svc, knn, dtc]
for model_ in models:
    print(model_.get_params())

# Data_set and scaling
scaler = StandardScaler()
X, y = load_breast_cancer(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape)


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


def get_rejected_data(x, y):
    r_data = []
    for i in x:
        r_data.append(y[i])
    return np.array(r_data)


# classes labels
proto_classes = np.array([0, 1])

# object of Hybrid class from prosemble
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')

pred1 = svc.predict(X_test)
pred2 = knn.predict(X_test)
pred3 = dtc.predict(X_test)

# confidence of prediction using the svc,knn and dtc models respectively
sec1 = get_posterior(x=X_test, y_=pred1, z_=svc)
sec2 = get_posterior(x=X_test, y_=pred2, z_=knn)
sec3 = get_posterior(x=X_test, y_=pred3, z_=dtc)
all_pred = [pred1, pred2, pred3]
all_sec = [sec1, sec2, sec3]

# predicted labels from the ensemble using hard voting
prediction1 = ensemble.pred_prob(X_test, all_pred)
# predicted labels from the ensemble using soft voting
prediction2 = ensemble.pred_sprob(X_test, all_sec)
# confidence of the prediction using hard voting
hard_prob = ensemble.prob(X_test, all_pred)
# confidence of the prediction using soft voting
soft_prob = ensemble.sprob(X_test, all_sec)
# print(soft_prob)
print(get_ens_confidence(prediction2, soft_prob))

# object instance for class related confidence thresh-hold.
crt = ThreshT(predict_results=prediction1, reject_rate1=0.1)

# object instance for class responsible for sorting based on crt algorithm
protocert = ProtoCert(y_test=y_test)

# predicted labels from the ensemble along with the confidence.
d1 = get_ens_confidence(prediction2, soft_prob)

# class related confidence thresh-hold for Malignant
h1 = crt.thresh_new(d1=d1, protocert_1=protocert, j=0)
#  class related confidence thresh-hold for Benign
h2 = crt.thresh_new(d1=d1, protocert_1=protocert, j=1)

# object needed to sorting for final results
protocertt = ProtoCertt(y_test=y_test)

non_rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='>=', y__='l', l3=[0, 1])
index_non_rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='>=', y__='i', l3=[0, 1])
true_labelsN = protocertt.thresh_y_test(x=index_non_rejected_labels)

# accuracy of model without rejection
accuracy1 = accuracy_score(y_true=y_test, y_pred=prediction2)

# accuracy of model with rejection max accepted rejection rate of 10% of test set w.r.t crt
accuracy = accuracy_score(y_true=true_labelsN, y_pred=non_rejected_labels)

# summary of the performance.
print([accuracy1, accuracy])

# Access rejected emsemble predicted labels
rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='<', y__='l', l3=[0, 1])

# Access index of rejected labels
index_rejected_labels = protocertt.thresh_function(x=d1, y=[h1, h2], y_='<', y__='i', l3=[0, 1])

# Access rejected data points in the test data.
print(get_rejected_data(x=index_rejected_labels, y=X_test))
