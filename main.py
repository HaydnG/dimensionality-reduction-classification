import classification
import data
import reduction
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.model_selection import train_test_split


DataList = data.load_data()

for do in DataList:
    print("\n\n#### " + do.name + " ####")

    do.xTrainingData, do.xTestData, do.yTrainingData, do.yTestData = reduction.prepareData(do.x, do.y)
    for classifier in classification.classificationAlgorithms:
        temp_score, elapsedTime = classifier.execute(do.xTrainingData, do.xTestData,
                                                     do.yTrainingData,
                                                     do.yTestData)
        do.addClassifierScore(classifier.name, temp_score, elapsedTime)

    for method in reduction.reductionAlgorithms:

        dataset = do.newReducedDataSet(method.name)
        for dimension in range(do.maxDimensionalReduction, 0, -1):

            if method.capByClasses and dimension > do.classes - 1:
                reducedData = dataset.addReducedData([], [], [], dimension, 0)
                for classifier in classification.classificationAlgorithms:
                    reducedData.addClassifierScore(classifier.name, 0, 0)

                continue

            reducedData = method.execute(dimension, do.x, do.y, dataset)

            for classifier in classification.classificationAlgorithms:
                temp_score, elapsedTime = classifier.execute(reducedData.xTrainingData, reducedData.xTestData, dataset.yTrainingData,
                                                             dataset.yTestData)
                reducedData.addClassifierScore(classifier.name, temp_score, elapsedTime)

                # print(method.name + " with (", classifier.name, ") classifier: ReductionScore (", temp_score,
                #       ") Dimension: (", dimension, "), classes: (",
                #       do.classes, ")")

            print('.', end='')
        print("")

    do.createGraph()
    do.createSpreadSheet()

data.workbook.close()

