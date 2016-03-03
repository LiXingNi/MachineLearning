from math import log
import helpFunc as hF

def calShannonEntropy(dataSet):
    setNum = len(dataSet)
    valueDct = {}
    entropy = 0.0

    for featVec in dataSet:
        currentVal = featVec[-1]
        if currentVal not in valueDct.keys():
            valueDct[currentVal] = 0
        valueDct[currentVal] += 1

    for value in valueDct:
        prob = float(valueDct[value]) / setNum
        entropy -= prob * log(prob,2)

    return entropy


def chooseBestFeatureShannon(dataSet):
    featureNum = len(dataSet[0]) - 1
    dataSetNum = len(dataSet)
    bestEntropy = 0.0;bestFeature = -1
    baseEntropy = calShannonEntropy(dataSet)

    for axis in range(featureNum):
        featValLis = [line[axis] for line in dataSet]
        featValSet = set(featValLis)

        currEntropy = 0.0

        for featVal in featValSet:
            subDataSet = hF.splitDataSet(dataSet,axis,featVal)
            prob = float(len(subDataSet)) / dataSetNum

            currEntropy += prob * calShannonEntropy(subDataSet)

        infoEntropy = baseEntropy - currEntropy
        if infoEntropy - bestEntropy > hF.G_Accuracy:
            #print('*********',infoEntropy,'\t',bestEntropy)
            bestEntropy = infoEntropy
            bestFeature = axis


    return bestFeature



#################################################