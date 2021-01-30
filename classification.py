from sklearn import neighbors
from sklearn import metrics
from timeit import default_timer as timer


class ClassificationMethod:
    def __init__(self, name, method):
        self.name = name
        self.method = method

    def execute(self, xTrainingData, xTestData, yTrainingData, yTestData):
        start = timer()
        yTestPredictedData = self.method(xTrainingData, xTestData, yTrainingData, yTestData)
        end = timer()



        return metrics.accuracy_score(yTestData, yTestPredictedData), (end - start) * 1000


classificationAlgorithms: ClassificationMethod = []


classificationAlgorithms.append(ClassificationMethod("KNeighbors",
                                           lambda xTrainingData, xTestData, yTrainingData, yTestData:
                                           neighbors.KNeighborsClassifier().fit(xTrainingData, yTrainingData).predict(xTestData)))
