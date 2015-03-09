__author__ = 'DCFURLA'

import numpy as np
import matplotlib.pyplot as plt

def PlotError(errors):
    plt.figure()
    plt.plot(errors)
    plt.title('Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

def GradientDescent(ComputeGradient, features, target, alpha, initCoefEstimate):
    pastCoefs = initCoefEstimate
    currentCoefs = initCoefEstimate
    init = False
    errors = []
    iter = 0
    while not init or not np.allclose(pastCoefs, currentCoefs, 1e-5):
        init = True
        [cost, gradient] = ComputeGradient(features, target, alpha, currentCoefs)
        errors.append(abs(cost))
        pastCoefs = np.copy(currentCoefs)
        currentCoefs += gradient
        if iter % 1000 == 0:
            print("Cost:{0} Gradient:{1} Current Coefs:{2}".format(cost, gradient, currentCoefs))
        iter += 1
    return errors, currentCoefs