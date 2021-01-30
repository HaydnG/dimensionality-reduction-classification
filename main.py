import sklearn
from sklearn.manifold import LocallyLinearEmbedding, Isomap
import data, classification, reduction

def main():
    print("Hello World!")

if __name__ == "__main__":
    main()

DataList = data.load_data()


for do in DataList:
    print("\n\n#### " + do.name + " ####")

    for method in reduction.reductionAlgorithms:

        dataset = do.newReducedDataSet(method.name)
        for dimension in range(do.maxDimensionalReduction, 0, -1):

            if method.capByClasses and dimension > do.classes - 1:
                data = dataset.addReducedData([], [], [], dimension, 0)
                for classifier in classification.classificationAlgorithms:
                    data.addClassifierScore(classifier.name, 0, 0)

                continue

            data = method.execute(dimension, do.x, do.y, dataset)

            for classifier in classification.classificationAlgorithms:
                temp_score, elapsedTime = classifier.execute(data.xTrainingData, data.xTestData, dataset.yTrainingData, dataset.yTestData)
                data.addClassifierScore(classifier.name, temp_score, elapsedTime)

                print(method.name + " with (", classifier.name, ") classifier: ReductionScore (", temp_score,
                      ") Dimension: (", dimension, "), classes: (",
                      do.classes, ")")

    do.createGraph()
