import sklearn
from sklearn.manifold import LocallyLinearEmbedding, Isomap
import data
import classification

DataList = data.load_data()

for do in DataList:
    print("\n\n#### " + do.name + " ####")
    graph = do.newGraph("KNeighbor")

    #Original score, no reduction

    graph.firstDataPoint = data.IterationData(do.x, do.y, do.dimensions)

    score = classification.apply_kneighbors_classifier(graph.firstDataPoint)
    print("Test sample size (",len(do.expect),"), Training size (", len(do.trained),")")
    print("BaseScore (",do.dimensions,"dimensions): ", score)


    # Run dimensionality reduction on every dimension, and find best score.
    algorithm = do.newAlgorithm("LocallyLinearEmbedding")
    while algorithm.dimensionIterator > 1:
        embedding = LocallyLinearEmbedding(n_components=algorithm.dimensionIterator)

        reducedX = embedding.fit_transform(do.x)

        point = do.newDataPoint(reducedX, do.y, algorithm.dimensionIterator)

        temp_score = classification.apply_kneighbors_classifier(point)

        if temp_score > algorithm.topScore:
            algorithm.topScore = temp_score
            algorithm.topDimensions = algorithm.dimensionIterator

        algorithm.dimensionIterator -=1

    print(algorithm.name + ": ReductionScore (", algorithm.topDimensions, "dimensions) : ", algorithm.topScore)
        # Run dimensionality reduction on every dimension, and find best score.
    algorithm = do.newAlgorithm("ISOMAP")
    while algorithm.dimensionIterator > 1:
        iso = Isomap(n_neighbors=6, n_components=algorithm.dimensionIterator)
        iso.fit(do.x)
        reducedX = iso.transform(do.x)

        point = do.newDataPoint(reducedX, do.y, algorithm.dimensionIterator)

        temp_score = classification.apply_kneighbors_classifier(point)

        if temp_score > algorithm.topScore:
            algorithm.topScore = temp_score
            algorithm.topDimensions = do.dimensionIterator

        algorithm.dimensionIterator -= 1

    do.createGraphs()

    print(algorithm.name + ": ReductionScore (",algorithm.topDimensions,"dimensions) : ", algorithm.topScore)



