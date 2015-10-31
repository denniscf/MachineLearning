import numpy as np
from KNN import KNN

if __name__ == '__main__':
    X = np.random.random([10, 2])
    y = np.random.random([10, 1]) > 0.5
    model = KNN(X, y)
    model.fit([0.5, 0.5])