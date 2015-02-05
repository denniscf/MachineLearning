__author__ = 'DCFURLA'
import numpy as np

def MaxValueScaling(features):
    maxPerColumn = np.max(features, axis=0)
    return features/maxPerColumn

def MeanNormalization(features):
    stdPerColumn = np.std(features, axis=0)
    meanPerColumn = np.mean(features, axis=0)
    return (features - meanPerColumn)/stdPerColumn