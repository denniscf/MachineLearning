__author__ = 'DCFURLA'
import unittest as ut
from KNN import KNN
import numpy as np

class KNN_test(ut.TestCase):
    def testKNN(self):
        X = np.array([[1, 1], [0, 0], [2, 1], [1, 2], [4, 4], [6,6]])
        y = np.array([0, 0, 0, 0, 1, 1])

        model = KNN(X, y)
        computedClass = model.fit([0.5, 0.5])
        trueClass = 0
        self.failIf(computedClass != trueClass)

        model = KNN(X, y)
        computedClass = model.fit([5, 5])
        trueClass = 1
        self.failIf(computedClass != trueClass)
