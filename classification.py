from sklearn import neighbors
from sklearn import metrics


class ClassificationMethod:
    def __init__(self, name, method):
        self.name = name
        self.method = method

    def execute(self, xTrainingData, xTestData, yTrainingData, yTestData):
        yTestPredictedData = self.method(xTrainingData, xTestData, yTrainingData, yTestData)

        return metrics.accuracy_score(yTestData, yTestPredictedData)


classificationAlgorithms: ClassificationMethod = []


classificationAlgorithms.append(ClassificationMethod("KNeighbors",
                                           lambda xTrainingData, xTestData, yTrainingData, yTestData:
                                           neighbors.KNeighborsClassifier().fit(xTrainingData, yTrainingData).predict(xTestData)))
