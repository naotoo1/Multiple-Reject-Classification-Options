import numpy as np
from sklearn.metrics import accuracy_score


def accuracy_rate1(x, y):
    """

    :param x: True labels
    :param y: predicted labels
    :return:  accuracy score
    """
    return accuracy_score(x, y)


def rejection_rate(x, y):
    """

    :param x: predicted labels below the thresh-hold confidence
    :param y: predicted labels regardless of thresh-hold confidence
    :return: rejection rate
    """
    z = len(x) / len(y)
    return z


class ThreshT:
    """
    Class Related Thresholds for multiple reject classifications
    :params
    Y_test : array: labels of the test_set
    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes
    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set
    reject_rate1: float: maximum rejection rate to be considered in the optimal search.

    """

    def __init__(self, predict_results, reject_rate1):
        self.predict_results = predict_results
        self.rejection_rate1 = reject_rate1

    def threshh(self, d1, protocert_1, j):
        """

        :param d1: The computed classification labels securities
        :param protocert_1: Class needed to do the sorting for the class_label_security.
        :param j: class_label under consideration for the optimal search
        :return:
        optimised list of all class related threshold, accuracy and rejection rate.
        """
        should_continue = True
        empty1 = []
        empty2 = []
        empty3 = []
        # y = thresh_hold
        j_ = 0
        for i in self.predict_results:
            if i == j:
                j_ += 1
        y = 1 / j_

        while should_continue:
            y = y + 0.01
            # list of predicted labels whose confidence is less than the thresh-hold for a given class
            index_listgl = protocert_1.thresh_function(x=d1, y=y, y_='<', y__='l', y___=j)
            # list of predicted labels regardless of the confidence thresh-hold for a given class
            index_listgl_ = protocert_1.thresh_function(x=d1, y=0, y_='>', y__='l', y___=j)
            # list containing index of predicted labels greater than or equal a given confidence thresh-hold
            index_listgi = protocert_1.thresh_function(x=d1, y=y, y_='>=', y__='i', y___=j)
            # list containing predicted labels greater than or equal to a given confidence thresh-hold
            index_listgi_ = protocert_1.thresh_function(x=d1, y=y, y_='>=', y__='l', y___=j)
            # computes the rejection rate based on the confidence thresh-hold
            z = rejection_rate(index_listgl, index_listgl_)
            # Actual labels of non-rejected classifications using the using their indexes
            true_labels = protocert_1.thresh_y_test(x=index_listgi)
            # computes accuracy of non-rejected classification
            z_ = accuracy_rate1(true_labels, index_listgi_)
            empty1.append([y, z_, z])
            empty2.append(z)
            empty3.append(z_)
            if z > self.rejection_rate1:
                should_continue = False
        return empty1[:-1], empty2[:-1], empty3[:-1]

    def thresh_new(self, d1, protocert_1, j):
        """
        :param d1: he computed classification labels securities
        :param protocert_1: Class needed to do the sorting for the class_label_security.
        :param j: class_label under consideration for the optimal search
        :return:
        optimised class related thresh-hold security. The thresh-hold at which we have
        minimum rejection and max accuracy
        """
        empty = []
        y = self.threshh(d1=d1, protocert_1=protocert_1, j=j)
        for i in range(len(y[1])):
            z = y[1][i]
            z_ = y[2][i]
            with np.errstate(divide='ignore', invalid='ignore'):
                z__ = z_ // z
                empty.append(z__)
        d = max([(i, v) for i, v in enumerate(empty)])
        return y[0][d[0]][0]
