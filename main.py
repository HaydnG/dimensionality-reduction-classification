import sklearn
from sklearn.manifold import LocallyLinearEmbedding, Isomap
import data, classification, reduction
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def main():
    print("Hello World!")

if __name__ == "__main__":
    main()

DataList = data.load_data()
testDataPercent = 0.30
selectionSeed = 3

for do in DataList:
    print("\n\n#### " + do.name + " ####")

    for method in reduction.reductionAlgorithms:

        dataset = do.newReducedDataSet(method.name)
        for dimension in range(do.maxDimensionalReduction, 0, -1):

            if method.capByClasses and dimension > do.classes - 1:
                continue

            reducedData = method.execute(dimension, do.x, do.y)

            xTrainingData, xTestData, dataset.yTrainingData, dataset.yTestData = train_test_split(reducedData, do.y,
                                                                                              test_size=testDataPercent,
                                                                                              random_state=selectionSeed)  ##random_state=2 data seed
            xTrainingData = preprocessing.scale(xTrainingData)
            xTestData = preprocessing.scale(xTestData)

            data = dataset.addReducedData(reducedData, xTrainingData, xTestData, dimension)

            for classifier in classification.classificationAlgorithms:
                temp_score = classifier.execute(xTrainingData, xTestData, dataset.yTrainingData, dataset.yTestData)
                print(method.name + " with (",classifier.name,") classifier: ReductionScore (", temp_score, ") Dimension: (", dimension, "), classes: (",
                      do.classes, ")")
                data.addClassifierScore(classifier.name, temp_score)

    do.createGraph()
