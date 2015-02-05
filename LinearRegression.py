__author__ = 'DCFURLA'

import numpy as np
import numpy.linalg as la
import GenPoints as gp
import GradientDescent as gd

def NormalEquation(features, target):
    pinvFeatures = la.pinv(np.dot(features.transpose(), features))
    XtTarget = np.dot(features.transpose(), target)
    x = np.dot(pinvFeatures, XtTarget)
    return x

def Hypothesis(coefs, features):
    return np.dot(features, coefs.transpose())

def ComputeGradient(features, target, alpha, coefs):
    nExamples = features.shape[0]
    h = Hypothesis(coefs, features)
    cost = (1.0/(2*nExamples))*np.sum((h - target)**2) #sum of mean squared errors
    diffCost = -(1.0/nExamples)*np.dot(features.transpose(), (h - target))
    gradient = alpha*diffCost #gradient for all tethas. xo = 1
    return cost, gradient

def RunLinRegressNormalFunction(x, y):
    #b == 0 & a == 1
    lineCoefs = NormalEquation(x, y)
    gp.Plot2DResults('Normal Function', x, y, lineCoefs)

def RunLinRegressGradientDescent(x, y):
    learningRate = 0.02
    initCoefEstimate = np.array([1.0, 1.0])
    coefs = gd.GradientDescent(ComputeGradient, x, y, learningRate, initCoefEstimate)
    gp.Plot2DResults('Gradient Descent', x, y, coefs)

if __name__ == '__main__':
    coeffs = np.array([0, 1])
    [x, y] = gp.GenerateLinear2DPoints(coeffs, 50)
    RunLinRegressGradientDescent(x, y)
    RunLinRegressNormalFunction(x, y)
