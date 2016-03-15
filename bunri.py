# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

data = iris.data[0:100]
target = iris.target[0:100]

train_x, test_x, train_y, test_y = cross_validation.train_test_split(data, target, test_size=0.2)

w = np.linalg.inv(train_x.T.dot(train_x)).dot(train_x.T).dot(train_y)
pred_y = np.array([1 if w.dot(x) > 0 else -1 for x in test_x])


# テストデータに対する正答率
print metrics.accuracy_score(test_y, pred_y)


