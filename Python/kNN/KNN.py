__author__ = 'DCFURLA'

import numpy as np

class KNN:
    def __init__(self, X, y, k=3, distMetric='euclidean'):
        self._k = k
        self._X = X
        self._y = y
        self._distMetric = distMetric
        self._distFuncs = self.BuildDistFunctions()

    def BuildDistFunctions(self):
        distFuncs = dict()
        distFuncs['euclidean'] = lambda p1, p2: np.sqrt(np.sum(np.power(p1 - p2, 2)))
        return distFuncs

    def ComputeDistances(self, X, dataPoint):
        distances = []
        distFunc = self._distFuncs[self._distMetric]
        for x in X:
            distances.append(distFunc(x, dataPoint))
        return distances

    def GetKNN(self, dists, k):
        idxs = np.argsort(dists)
        sortedX = self._X[idxs]
        sortedy = self._y[idxs]
        return sortedX[0:k], sortedy[0:k]

    def fit(self, dataPoint):
        dists = self.ComputeDistances(self._X, dataPoint)
        [X, y] = self.GetKNN(dists, self._k)
        uniqueClasses = np.unique(y)

        count = [0]*len(uniqueClasses)
        for c in range(0, len(uniqueClasses)):
            count[c] = np.sum(c == y)
        idxs = np.argsort(count)[::-1]
        return y[idxs[-1]]







