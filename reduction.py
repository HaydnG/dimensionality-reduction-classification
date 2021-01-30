from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from timeit import default_timer as timer

testDataPercent = 0.30
selectionSeed = 3

class ReductionMethod:
    def __init__(self, capByClasses, name, method):
        self.name = name
        self.method = method
        self.capByClasses = capByClasses

    def execute(self, dimension, x, y, dataset):
        start = timer()
        reducedData = self.method(dimension, x, y)
        end = timer()


        xTrainingData, xTestData, dataset.yTrainingData, dataset.yTestData = train_test_split(reducedData, y,
                                                                                              test_size=testDataPercent,
                                                                                              random_state=selectionSeed)  ##random_state=2 data seed
        xTrainingData = preprocessing.scale(xTrainingData)
        xTestData = preprocessing.scale(xTestData)

        return dataset.addReducedData(reducedData, xTrainingData, xTestData, dimension, (end - start) * 1000)




reductionAlgorithms: ReductionMethod = []



reductionAlgorithms.append(ReductionMethod(False, "LocallyLinearEmbedding",
                                           lambda dimensions, x, y:
                                           LocallyLinearEmbedding(n_components=dimensions).fit_transform(x)))


reductionAlgorithms.append(ReductionMethod(True, "LDA",
                                           lambda dimensions, x, y:
                                           LDA(n_components=dimensions).fit_transform(x, y)))

reductionAlgorithms.append(ReductionMethod(False, "Isomap",
                                           lambda dimensions, x, y:
                                           Isomap(n_components=dimensions).fit_transform(x)))

