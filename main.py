import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from pylab import rcParams, matplotlib
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data

np.set_printoptions(precision=4, suppress=True)
rcParams['figure.figsize'] = 7, 4
plt.style.use('seaborn-whitegrid')


def apply_kneighbors_classifier(do: data.DataObject, test_size: float,
                                selection_seed: int):

    x_train, x_test, y_train, y_test = train_test_split(do.x, do.y, test_size=test_size,
                                                        random_state=selection_seed)  ##random_state=2 data seed

    clf = neighbors.KNeighborsClassifier()

    clf.fit(x_train, y_train)

    p.expect = y_test
    p.predict = clf.predict(x_test)


DataList = data.load_data()

print("Data Loaded")

for p in DataList:
    print("#### " + p.name + " ####")

    apply_kneighbors_classifier(p, .30, 3)
    print(metrics.classification_report(p.expect, p.predict))

    embedding = LocallyLinearEmbedding(n_components=3)
    p.x = embedding.fit_transform(p.x)

    print("### With LLE ###")
    apply_kneighbors_classifier(p, .30, 3)
    print(metrics.classification_report(p.expect, p.predict))


