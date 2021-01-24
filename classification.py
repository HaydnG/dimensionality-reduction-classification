from sklearn import neighbors
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

import data

testDataPercent = 0.30
selectionSeed = 3

def apply_kneighbors_classifier(ID: data.IterationData):

    ID.xTrainingData, ID.xTestData, ID.yTrainingData, ID.yTestData = train_test_split(ID.xData, ID.yData, test_size=testDataPercent,
                                                        random_state=selectionSeed)  ##random_state=2 data seed


    ID.xTrainingData = preprocessing.scale(ID.xTrainingData)
    ID.xTestData = preprocessing.scale(ID.xTestData)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(ID.xTrainingData, ID.yTrainingData)


    ID.yTestPredictedData = clf.predict(ID.xTestData)
    ID.score = metrics.accuracy_score(ID.yTestData, ID.yTestPredictedData)


    return ID.score