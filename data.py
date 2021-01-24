import inline as inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=4, suppress=True)
plt.style.use('seaborn-whitegrid')



pd.set_option("display.max.columns", None)
from sklearn import metrics
GraphCount = 0

# Converts string data to enumerated data
def enumerate(csv, label):
    csv = csv.replace('?', np.nan)
    csv = csv.dropna()

    le = preprocessing.LabelEncoder()
    le.fit(csv[label])
    csv[label] = le.transform(csv[label])

    return csv


# Takes in the data, and a list of labels to apply enumeration
def enumerate_data(csv, labels):
    for l in labels:
        csv = enumerate(csv, l)

    return csv


# Enumerates all columns in dataset
def enumerate_all(csv):
    for l in csv.columns:
        csv = enumerate(csv, l)
    return csv

class IterationData:
    def __init__(self, xData, yData, dimension):
        self.xData = xData
        self.yData = yData
        self.dimension = dimension

        self.xTrainingData = []
        self.yTrainingData = []

        self.xTestData = []
        self.yTestData = []
        self.yTestPredictedData = []

        self.score = 0

class Algorithm:
    def __init__(self, name):
        self.name = name
        self.dataPoints: IterationData = []
        self.topScore = 0
        self.topDimensions = 0
        self.dimensionIterator = 0

class GraphDataBundle:

    def __init__(self, name):
        self.name = name
        self.firstDataPoint: IterationData
        self.algorithms: Algorithm = []

class DataObject:

    def createGraphs(self):
        global GraphCount

        for gdb in self.graphDataBundles:
            fig = plt.figure(GraphCount)

            scoreData = []
            for algo in gdb.algorithms:
                scores = [dp.score for dp in algo.dataPoints]
                scores.insert(0, gdb.firstDataPoint.score)

                dimensions = [dp.dimension for dp in algo.dataPoints]
                dimensions.insert(0, gdb.firstDataPoint.dimension)

                scoreData.append(scores)

            df = pd.DataFrame(np.c_[scoreData[0], scoreData[1]], index=np.arange(0, self.graphLabelMax, 1).tolist(),
                              columns=[algo.name for algo in gdb.algorithms])

            ax = df.plot.bar()

            lines, labels = ax.get_legend_handles_labels()

            plt.legend(lines, labels, title='Reduction Algorithm',
                       bbox_to_anchor=(0.5, -0.3), loc="lower center",
                       ncol=2, borderaxespad=0.)
            plt.subplots_adjust(wspace=2)
            plt.title(
                self.name + " Test sample size (" + str(len(gdb.algorithms[0].dataPoints[0].xTestData)) + "), Training size (" + str(
                    len(gdb.algorithms[0].dataPoints[0].xTrainingData)) + ")",
                loc='right')
            plt.title(gdb.name, loc='left')
            plt.ylabel("Prediction accuracy")
            plt.xlabel("Number of dimensions")
            plt.margins(y=0)

            plt.xticks(list(range(0, self.graphLabelMax)), dimensions)

            plt.savefig('graphs/' + self.name + '.png', bbox_inches='tight')
            plt.show()
            GraphCount += 1

    def newDataPoint(self, xData, yData, dimension):
        id = IterationData(xData, yData, dimension)

        self.graphDataBundles[-1].algorithms[-1].dataPoints.append(id)
        return id

    def newGraph(self, name):
        self.graphDataBundles.append(GraphDataBundle(name))

        return  self.graphDataBundles[-1]

    def newAlgorithm(self, name):
        algo = Algorithm(name)

        self.graphDataBundles[-1].algorithms.append(algo)

        algo.dimensionIterator = self.dimensions - 1
        if algo.dimensionIterator > 26:
            algo.dimensionIterator = 26


        return  algo

    def __init__(self, name, data):
        self.name = name

        data = data.replace('?', np.nan)
        data = data.dropna()
        datacopy = data.copy(deep=True)

        self.x = datacopy.drop(['Class identifier'],  axis=1)
        self.dimensions = len(data.columns)

        self.dimensionIterator = self.dimensions - 1
        if self.dimensionIterator > 26:
            self.dimensionIterator = 26
        self.graphLabelMax = self.dimensionIterator

        self.y = data['Class identifier']

        self.trained = []
        self.expect = []
        self.predict = []
        self.reduced_x = []


        self.graphDataBundles: GraphDataBundle = []




def load_data():
    DataList = []

    # Wine Data
    csv = pd.read_csv('data/wine.data')
    #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
    csv.columns = ['Class identifier', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

    DataList.append(DataObject('Wine', csv))

    # Cancer Data
    csv = pd.read_csv('data/breast-cancer.data')
    #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')
    csv.columns = ['Class identifier', 'Age', 'Menopause', 'Tumor-size', 'Inv-nodes', 'Node-caps', 'Deg-malig',
                   'Breast', 'Breast-Quad', 'Irradiat']
    csv = enumerate_all(csv)
    DataList.append(DataObject('Cancer', csv))

    # Hepatitis Data
    csv = pd.read_csv('data/hepatitis.data')
    #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data')
    csv.columns = ['Class identifier', 'Age', 'Sex', 'Steroid', 'AntiVirals', 'Fatigue', 'Malaise', 'Anorexia',
                   'Liver Big', 'Liver Firm', 'Spleen Palpable',
                   'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SGOT', 'Albumin', 'Protime',
                   'Histology']
    csv = enumerate_data(csv, ['Sex', 'Steroid', 'AntiVirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 'Liver Firm',
                         'Spleen Palpable',
                         'Spiders', 'Ascites', 'Varices', 'Histology'])
    DataList.append(DataObject('Hepatitis', csv))

    # Glass data
    csv = pd.read_csv('data/glass.data')
    #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
    csv.columns = ['ID', 'Refractive Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium',
                   'Barium', 'Iron', 'Class identifier']
    DataList.append(DataObject('Glass', csv))

    # Parkinsons data
    csv = pd.read_csv('data/parkinsons.data')
    # #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')
    csv.columns = ['Name', 'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                   'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
                   'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
                   'Class identifier', 'RPDE', 'DFA', 'spread1',
                   'spread2', 'D2', 'PPE']
    del csv['Name']

    DataList.append(DataObject('Parkinsons', csv))

    # Lung Cancer Data
    csv = pd.read_csv('data/lung-cancer.data')
    #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data')
    names = ['Class identifier']
    names = names + list(range(0,56))

    csv.columns = names
    DataList.append(DataObject('Lung cancer', csv))

    # Echocardiogram data
    # csv = pd.read_csv('data/echocardiogram.data',error_bad_lines=False)
    # #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data',error_bad_lines=False)
    # csv.columns = ['Survival', 'Still-alive', 'Age-at-heart-attack', 'Pericardial-effusion', 'Fractional-shortening',
    #                'EPSS', 'LVDD', 'Wall-motion-score',
    #                'Wall-motion-index', 'Mult', 'Name', 'Group', 'Alive-at-1']
    # del csv['Name']
    # del csv['Group']
    # # graph - https://realpython.com/pandas-plot-python/
    # #csv['Alive-at-1'].value_counts().plot(kind='bar').set_title('Echocardiogram Outcome')
    # #csv.head()
    # #plt.show()
    # DataList.append(DataObject('Echocardiogram', csv, [0, 10], 10))




    return DataList
