import inline as inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import classification
import matplotlib.patheffects as pe
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import xlsxwriter
import os

if os.path.exists("ReductionData.xlsx"):
  os.remove("ReductionData.xlsx")
workbook = xlsxwriter.Workbook('ReductionData.xlsx')

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset



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

class ReducedData:
    def __init__(self, xData, xTrainingData , xTestData, dimension, elapsedTime):
        self.xData = xData
        self.xTrainingData = xTrainingData
        self.xTestData = xTestData
        self.dimension = dimension
        self.classifierScore = {}
        self.classifierTime = {}
        self.elapsedTime = elapsedTime

    def addClassifierScore(self, name, score, elapsedTime):
        self.classifierScore[name] = score
        self.classifierTime[name] = elapsedTime


class ReducedDataSet:
    def __init__(self, name):
        self.name = name
        self.reducedData: ReducedData = []
        self.yTrainingData = None
        self.yTestData = None

    def addReducedData(self, xData, xTrainingData , xTestData, Dimension, elapsedTime):
        self.reducedData.append(ReducedData(xData, xTrainingData , xTestData, Dimension, elapsedTime))
        return self.reducedData[-1]


class DataObject:

    def __init__(self, name, data):
        self.name = name


        data = data.replace('?', np.nan)
        data = data.dropna()
        datacopy = data.copy(deep=True)

        self.x = datacopy.drop(['Class identifier'],  axis=1)
        self.dimensions = len(data.columns) - 1

        self.maxDimensionalReduction = self.dimensions - 1
        if self.maxDimensionalReduction > 26:
            self.maxDimensionalReduction = 26

        self.y = data['Class identifier']
        self.classes = self.y.nunique()
        self.yTrainingData = None
        self.yTestData = None
        self.xTrainingData = None
        self.xTestData = None
        self.reducedDataSets: ReducedDataSet = []
        self.classifierScore = {}
        self.classifierTime = {}

    def addClassifierScore(self, name, score, elapsedTime):
        self.classifierScore[name] = score
        self.classifierTime[name] = elapsedTime

    def newReducedDataSet(self, name):
        self.reducedDataSets.append(ReducedDataSet(name))
        return self.reducedDataSets[-1]

    def createSpreadSheet(self):
        worksheet = workbook.add_worksheet(self.name)
        worksheet.set_column(0, (len(self.reducedDataSets) * 2) + 1, 20)

        row = 0
        col = 0

        worksheet.write(row, col, "Result without reduction")
        row=1
        worksheet.write(row, col, "Classification Algorithm")
        worksheet.write(row, col + 1, "Dimensions")
        worksheet.write(row, col + 2, "Score")
        worksheet.write(row, col + 3, "Classification Time")
        row += 1
        for classifier in classification.classificationAlgorithms:
            worksheet.write(row, col, classifier.name)
            worksheet.write(row, col + 1, self.dimensions)
            worksheet.write(row, col + 2, self.classifierScore[classifier.name])
            worksheet.write(row , col + 3, self.classifierTime[classifier.name])
            row+=1

        worksheet.write(row + 3, col, "Reduction scores")
        row +=4

        for classifier in classification.classificationAlgorithms:
            col = 0
            worksheet.write(row, col, classifier.name + " Classifier")
            row += 1
            rowOffset = row
            row+=1

            for dimension in [ds.dimension for ds in self.reducedDataSets[0].reducedData]:
                worksheet.write(row, col, dimension)
                row+=1
            row = rowOffset

            colOffset = 1

            for datasets in self.reducedDataSets:
                col = colOffset
                row=rowOffset
                worksheet.write(row, col, datasets.name + " Score")
                worksheet.write(row, col + 1, datasets.name + " Time")
                row+=1
                for data in datasets.reducedData:
                    col = colOffset
                    worksheet.write(row, col, data.classifierScore["KNeighbors"])
                    worksheet.write(row, col + 1, data.elapsedTime)
                    col = colOffset
                    row+=1
                colOffset+=2



    def createGraph(self):
        global GraphCount

        plt.figure(GraphCount)

        for classifier in classification.classificationAlgorithms:

            scoreData = []
            for datasets in self.reducedDataSets:
                scores = [ds.classifierScore[classifier.name] for ds in datasets.reducedData]
                scoreData.append(scores)

                dimensions = [ds.dimension for ds in datasets.reducedData]

            df = pd.DataFrame(np.c_[scoreData[0], scoreData[1], scoreData[2]], index=np.arange(0, self.maxDimensionalReduction, 1).tolist(),
                              columns=[rds.name for rds in self.reducedDataSets])

            ax = df.plot.bar()

            lines, labels = ax.get_legend_handles_labels()

            plt.legend(lines, labels, title='Reduction Algorithm',
                       bbox_to_anchor=(0, -0.3), loc="lower left",
                       ncol=2, borderaxespad=0.)
            plt.subplots_adjust(wspace=2)
            plt.title(
                self.name,
                loc='right')

            plt.title(classifier.name, loc='left')
            plt.ylabel("Prediction accuracy (bars)")
            plt.xlabel("Number of dimensions")
            plt.margins(y=0)

            plt.xticks(list(range(0, self.maxDimensionalReduction)), dimensions)

            ax2 = ax.twinx()
            for datasets in self.reducedDataSets:
                ax2.plot(list(range(0, self.maxDimensionalReduction)),
                                [ds.elapsedTime for ds in datasets.reducedData],marker='o',markersize=4, lw=1, markeredgecolor='black')

            # ax2.plot(list(range(0, self.maxDimensionalReduction)),
            #          [ds.classifierTime[classifier.name] for ds in datasets.reducedData], marker='*', markersize=5, lw=1,
            #          markeredgecolor='red')

            ax2.legend(handles=[Line2D([0], [0], marker='o', color='black', label='Reduction Time',
                                      markerfacecolor='red', markersize=10)],
                       bbox_to_anchor=(1, -0.3),title='Execution Time (ms)', loc="lower right",
                       ncol=1, borderaxespad=0.)


            plt.ylabel("Algorithm execution time (ms) (line)")
            ax2.set_ylim(bottom=0)

            plt.savefig('graphs/' + self.name + "_" + classifier.name + '.png', bbox_inches='tight')
            plt.show()
            GraphCount += 1



def load_data():
    DataList = []

    #https://archive.ics.uci.edu/ml/datasets/Poker+Hand
    # Poker Data
    csv = pd.read_csv('data/poker-hand.data')
    # csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data')
    csv.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class identifier']
    DataList.append(DataObject('Poker', csv))

    # Cancer Data
    csv = pd.read_csv('data/breast-cancer.data')
    # csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')
    csv.columns = ['Class identifier', 'Age', 'Menopause', 'Tumor-size', 'Inv-nodes', 'Node-caps', 'Deg-malig',
                   'Breast', 'Breast-Quad', 'Irradiat']
    csv = enumerate_all(csv)
    DataList.append(DataObject('Cancer', csv))

    # Wine Data
    csv = pd.read_csv('data/wine.data')
    #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
    csv.columns = ['Class identifier', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

    DataList.append(DataObject('Wine', csv))

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

    # Glass data
    csv = pd.read_csv('data/glass.data')
    # csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
    csv.columns = ['ID', 'Refractive Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium',
                   'Barium', 'Iron', 'Class identifier']
    DataList.append(DataObject('Glass', csv))



    # Hepatitis Data
    csv = pd.read_csv('data/hepatitis.data')
    # csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data')
    csv.columns = ['Class identifier', 'Age', 'Sex', 'Steroid', 'AntiVirals', 'Fatigue', 'Malaise', 'Anorexia',
                   'Liver Big', 'Liver Firm', 'Spleen Palpable',
                   'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SGOT', 'Albumin', 'Protime',
                   'Histology']
    csv = enumerate_data(csv,
                         ['Sex', 'Steroid', 'AntiVirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 'Liver Firm',
                          'Spleen Palpable',
                          'Spiders', 'Ascites', 'Varices', 'Histology'])
    DataList.append(DataObject('Hepatitis', csv))



    # # Lung Cancer Data
    # csv = pd.read_csv('data/lung-cancer.data')
    # # csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data')
    # names = ['Class identifier']
    # names = names + list(range(0, 56))
    # csv.columns = names
    # csv[3] = pd.to_numeric(csv[3], errors='coerce').fillna(0).astype(np.int64)
    # csv[37] = pd.to_numeric(csv[37], errors='coerce').fillna(0).astype(np.int64)
    # DataList.append(DataObject('Lung cancer', csv))

    # #Echocardiogram data
    # csv = pd.read_csv('data/echocardiogram.data',error_bad_lines=False)
    # #csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data',error_bad_lines=False)
    # csv.columns = ['Survival', 'Still-alive', 'Age-at-heart-attack', 'Pericardial-effusion', 'Fractional-shortening',
    #                'EPSS', 'LVDD', 'Wall-motion-score',
    #                'Wall-motion-index', 'Mult', 'Name', 'Group', 'Class identifier']
    # del csv['Name']
    # del csv['Group']
    # DataList.append(DataObject('Echocardiogram', csv))




    return DataList
