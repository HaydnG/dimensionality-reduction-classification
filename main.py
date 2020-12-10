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


def apply_kneighbors_classifier(X: any, Y: any, test_size: float,
                                selection_seed: int):

    X = preprocessing.scale(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=selection_seed)  ##random_state=2 data seed

    clf = neighbors.KNeighborsClassifier()

    clf.fit(x_train, y_train)

    p.expect = y_test
    p.predict = clf.predict(x_test)
    return clf.score(x_test, p.expect)


DataList = data.load_data()

print("Data Loaded")

for p in DataList:
    print("\n\n#### " + p.name + " ####")

    score = apply_kneighbors_classifier(p.x, p.y, .30, 3)
    print("Accuracy with (",len(p.expect),") predictions")
    print("BaseScore (",p.dimensions,"dimensions): ", score)
    #print(metrics.classification_report(p.expect, p.predict))

    dimension = p.dimensions - 1
    if dimension >26:
        dimension = 26

    top_score = 0
    best_dimensions = 0
    while dimension > 1:
        embedding = LocallyLinearEmbedding(n_components=dimension)
        p.red_x = embedding.fit_transform(p.x)

        temp_score = apply_kneighbors_classifier(p.red_x, p.y, .30, 3)
        if temp_score > top_score:
            top_score = temp_score
            best_dimensions = dimension

        dimension -=1

    print("ReductionScore (",best_dimensions,"dimensions) : ", top_score)

    #print(metrics.classification_report(p.expect, p.predict))


