from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class ReductionMethod:
    def __init__(self, capByClasses, name, method):
        self.name = name
        self.method = method
        self.capByClasses = capByClasses

    def execute(self, dimensions, x, y):
        return self.method(dimensions, x, y)




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

