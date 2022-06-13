"""Module to determine a unique and multiple reject options for improving classification reliability
of prototype and non prototype-based models."""


class ProtoCert:
    """
    Prototype certainty and overall model certainty
    :param
    x_test: array, shape=[num_data,num_features]
            Where num_data is the number of samples and num_features refers to the number of features.

    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes

    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set

    """

    def __init__(self, y_test):
        self.y_test = y_test

    def thresh_function(self, x, y, y_, y__, y___):
        """
        :param x: predicted labels with their corresponding securities
        :param y: float: security threshold
        :param y_: string: '>', '<' ,'>=' to indicate the threshold security
        :param y__: string: 's' for securities , 'i' for index of data point, 'l' for label, 'all' for list with
                (securities,indexes,labels)
        :param y___: class label under consideration (None for all labels)
        :return: List containing securities( greater than or less than a given security thresh-hold),
        """
        empty = []
        empty2 = []
        empty3 = []
        empty4 = []
        for i in range(len(x)):
            if y_ == '>' and x[i][1] > y and x[i][0] == y___:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '>' and x[i][1] > y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '>=' and x[i][1] >= y and x[i][0] == y___:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '>=' and x[i][1] >= y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '=' and x[i][1] == y and x[i][0] == y___:
                empty.append(x[i][1])
                empty3.append(x[i][0])
                empty2.append(i)
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '=' and x[i][1] == y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '<' and x[i][1] < y and x[i][0] == y___:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '<' and x[i][1] < y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
        if y__ == 'i':
            return empty2
        if y__ == 's':
            return empty
        if y__ == 'l':
            return empty3
        if y__ == 'all':
            return empty4

    def thresh_y_test(self, x):
        """

        :param x: thresh hold index list
        :return: labels with the thresh hold security
        """
        empty = []
        y = self.y_test
        for index in x:
            for i in range(len(y)):
                if i == index:
                    empty.append(y[i])
        return empty


if __name__ == '__main__':
    print('import module to use')
