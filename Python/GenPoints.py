__author__ = 'DCFURLA'
import numpy as np
import matplotlib.pyplot as plt

def GenerateLinear2DPoints(coeffs, numPoints):
    Xs = np.ones([numPoints, len(coeffs)])
    Xs[:, 1] = np.linspace(0, 10, numPoints)
    Ys = np.dot(coeffs, Xs.transpose())
    for i in range(numPoints):
        Ys[i] += np.random.random()*5 + (-1)**np.random.choice([1, 2])
    return Xs, Ys

def GenerateRandom2DPoints(coeffs, numPoints):
    Xs = np.ones([numPoints, len(coeffs)])
    Ys = np.ones([numPoints, 1])
    for i in range(numPoints):
        Xs[i,1] = np.random.random()
        Ys[i] = np.random.random()
    return Xs, Ys

def Plot2DRegressionLine(title, x, y, coefs):
    plt.figure()
    plt.scatter(x[:, 1], y)
    plt.title("a={0}, b={1} - {2}".format(coefs[1], coefs[0], title))
    plt.plot(x[:, 1], np.dot(x, coefs.transpose()))
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()

def PlotSurface(minVal, maxVal, coefs, mapFunc, h):
    nValues = 50
    Xs = np.ones([nValues, 2])
    Xs[:,1] = np.linspace(minVal, maxVal, nValues)
    Ys = np.linspace(minVal, maxVal, nValues)
    [X, Y] = np.meshgrid(Xs[:,1], Ys)

    dS = np.hstack([np.ones([X.size, 1]), X.reshape([X.size, 1])])
    dS = np.hstack([dS, Y.reshape([Y.size, 1])])
    dS = mapFunc(dS)
    Z = h(coefs, dS)
    plt.contour(X, Y, Z.reshape(X.shape))
    plt.colorbar()
    plt.draw()

