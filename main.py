import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pylab import rcParams, matplotlib
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data

np.set_printoptions(precision=4, suppress=True)
rcParams['figure.figsize'] = 7, 4
plt.style.use('seaborn-whitegrid')


def apply_kneighbors_classifier(data: any, dimensions_range: slice, class_index: int, test_size: float,
                                selection_seed: int):
    x_prime = data.iloc[:, dimensions_range[0]:dimensions_range[1]].values

    x = preprocessing.scale(x_prime)

    y = data.iloc[:, class_index].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=selection_seed)  ##random_state=2 data seed

    clf = neighbors.KNeighborsClassifier()

    clf.fit(x_train, y_train)

    y_expect = y_test
    y_pred = clf.predict(x_test)

    return metrics.classification_report(y_expect, y_pred)


DataList = data.load_data()

print("Data Loaded")

for p in DataList:
    print("#### " + p.name + " ####")
    print(apply_kneighbors_classifier(p.data, p.dimensions_range, p.class_index, .30, 2))
