__author__ = 'DCFURLA'

import numpy as np
import GenPoints as gp
import GradientDescent as gd
import matplotlib.pyplot as plt
import sys

def ComputeGradient(features, target, alpha, coefs):
    nExamples = features.shape[0]
    h = Hypothesis(coefs, features)
    cost = (1.0/nExamples)*np.sum(-target*np.log(h) - (1-target)*np.log(1-h))
    diffCost = -np.dot(features.transpose(), (h - target))
    gradient = alpha*diffCost #gradient for all tethas. xo = 1
    return cost, gradient

def RunLogRegressGradientDescent(x, y):
    learningRate = 0.1
    initCoefEstimate = np.ones([1,x.shape[1]])
    [errors, coefs] = gd.GradientDescent(ComputeGradient, x, y, learningRate, initCoefEstimate)
    return coefs

def Hypothesis(coefs, features):
    return (1.0/(1 + np.exp(-np.dot(features, coefs.transpose())))).reshape(-1)

def ClassifyPoints(x, y):
    return np.sqrt((x[:,1] - 0.5)**2 + (y[:,0] - 0.5)**2) <= 0.2

#Map the 2D points [x,y] to a higher dimension: [xo, x1, x2, x3, x4, x5] => [1, x, y, x*y, x*x, y*y]
def MapDimensions(dS):
    dS = np.hstack((dS, (dS[:,1]*dS[:,2]).reshape([dS.shape[0], 1])))
    dS = np.hstack((dS, (dS[:,1]*dS[:,1]).reshape([dS.shape[0], 1])))
    dS = np.hstack((dS, (dS[:,2]*dS[:,2]).reshape([dS.shape[0], 1])))
    return dS

if __name__ == '__main__':
    [x, y] = gp.GenerateRandom2DPoints([0,1], 200)
    target = ClassifyPoints(x, y)
    cols = ['w' if val else 'k' for val in target]

    dataSet = MapDimensions(np.hstack((x, y)))
    coefs = RunLogRegressGradientDescent(dataSet, target.astype(int))

    #plot results
    plt.scatter(x[:, 1], y, c=cols, s=40)
    gp.PlotSurface(0, 1, coefs, MapDimensions, Hypothesis)
    plt.show()

