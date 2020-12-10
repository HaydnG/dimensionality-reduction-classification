import inline as inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

pd.set_option("display.max.columns", None)
from sklearn import metrics


# Converts string data to enumerated data
def enumerate(csv, label):
    le = preprocessing.LabelEncoder()
    le.fit(csv[label])
    csv[label] = le.transform(csv[label])


# Takes in the data, and a list of labels to apply enumeration
def enumerate_data(csv, labels):
    for l in labels:
        enumerate(csv, l)


# Enumerates all columns in dataset
def enumerate_all(csv):
    for l in csv.columns:
        enumerate(csv, l)


class DataObject:
    expect = []
    predict = []

    def __init__(self, name, data, dimensions_range, class_index):
        self.name = name
        self.dimensions = dimensions_range[1]

        data = data.replace('?', np.nan)
        data = data.dropna()
        x_prime = data.iloc[:, dimensions_range[0]:dimensions_range[1]].values
        self.x = preprocessing.scale(x_prime)
        self.y = data.iloc[:, class_index].values




def load_data():
    DataList = []

    # Wine Data
    csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
    csv.columns = ['Class identifier', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
    DataList.append(DataObject('Wine', csv, [1, 14], 0))

    # Cancer Data
    csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')
    csv.columns = ['Class identifier', 'Age', 'Menopause', 'Tumor-size', 'Inv-nodes', 'Node-caps', 'Deg-malig',
                   'Breast', 'Breast-Quad', 'Irradiat']
    enumerate_all(csv)
    DataList.append(DataObject('Cancer', csv, [1, 9], 0))

    # Hepatitis Data
    csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data')
    csv.columns = ['Class identifier', 'Age', 'Sex', 'Steroid', 'AntiVirals', 'Fatigue', 'Malaise', 'Anorexia',
                   'Liver Big', 'Liver Firm', 'Spleen Palpable',
                   'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SGOT', 'Albumin', 'Protime',
                   'Histology']
    enumerate_data(csv, ['Sex', 'Steroid', 'AntiVirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 'Liver Firm',
                         'Spleen Palpable',
                         'Spiders', 'Ascites', 'Varices', 'Histology'])
    DataList.append(DataObject('Hepatitis', csv, [1, 20], 0))

    # Glass data
    csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
    csv.columns = ['ID', 'Refractive Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium',
                   'Barium', 'Iron', 'Class identifier']
    DataList.append(DataObject('Glass', csv, [1, 10], 10))

    # Lung Cancer Data
    csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data')
    DataList.append(DataObject('Lung cancer', csv, [1, 56], 0))

    # Echocardiogram data
    csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data',
                      error_bad_lines=False)
    csv.columns = ['Survival', 'Still-alive', 'Age-at-heart-attack', 'Pericardial-effusion', 'Fractional-shortening',
                   'EPSS', 'LVDD', 'Wall-motion-score',
                   'Wall-motion-index', 'Mult', 'Name', 'Group', 'Alive-at-1']
    del csv['Name']
    # graph - https://realpython.com/pandas-plot-python/
    # csv['Alive-at-1'].value_counts().plot(kind='bar').set_title('Echocardiogram Outcome')
    # csv.head()
    # plt.show()
    DataList.append(DataObject('Echocardiogram', csv, [0, 11], 11))

    return DataList
