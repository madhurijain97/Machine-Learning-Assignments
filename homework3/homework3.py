import numpy as np
import math
from Blackbox31 import blackbox31

numOfClasses, numberOfSamples, prior, noiseProbability = 3, 1000, 1/3, 0.000001 / 3
evidence = 1 / numberOfSamples
theoryList = [0, 0, 0]
allInputList = []

def calculateMean(data):
    allInputList.extend(data)
    return sum(allInputList)/len(allInputList)

def calculateVariance(mean):
    varianceList = [value - mean for value in allInputList]
    return (sum(map(lambda value : value * value, varianceList)))/len(allInputList)

def calculateNormalDistribution(data):
    mean = calculateMean(data)
    variance = calculateVariance(mean)
    return mean, variance

def calculateProbability(data, mean, variance):
    probabilityList = []
    for value in data:
        numerator = math.exp((-(float(value) - float(mean)) ** 2)/(2 * variance))
        denominator = math.sqrt(2 * math.pi) * math.sqrt(variance)
        probability = numerator / denominator
        probabilityList.append(probability)
    return probabilityList

def incrementalLearningFirstSample(likelihood, target):
    global theoryList
    prior = 1 / numOfClasses
    posterior = (prior * likelihood) / evidence
    for i in range(3):
        if target == i:
            theoryList[i] = posterior
        else:
            theoryList[i] = (prior * noiseProbability) / evidence

def incrementalLearningForOtherSamples(likelihood, target):
    global theoryList
    for i in range(3):
        if target == i:
            theoryList[i] = (theoryList[i] * likelihood) / evidence
        else:
            theoryList[i] = (theoryList[i] * noiseProbability) / evidence

def incrementalLearning(inputList, target, index):
    global prior
    mean, variance = calculateNormalDistribution(inputList)
    probabilityList = calculateProbability(inputList, mean, variance)
    likelihood = np.prod(np.array(probabilityList))
    if index == 1:
        incrementalLearningFirstSample(likelihood, target)
    else:
        incrementalLearningForOtherSamples(likelihood, target)

def calculateAccuracy(myPredictions, actualPredictions):
    count = 0
    for myPred, actualPred in zip(myPredictions, actualPredictions):
        if myPred != actualPred:
            print("Mismatch: ", myPred, actualPred)
            count += 1
    return (numberOfSamples - count) / numberOfSamples

def testPredictions(testingData, actualPrediction):
    resultList = []
    for inputList in testingData:
        probabilityList = calculateProbability(inputList, mean, variance)
        likelihood = np.prod(np.array(probabilityList))
        theoryA = (theoryList[0] * likelihood) / evidence
        theoryB = (theoryList[1] * likelihood) / evidence
        theoryC = (theoryList[2] * likelihood) / evidence
        maxValue = max(max(theoryA, theoryB), theoryC)
        if maxValue == theoryA:
            resultList.append(0)
        elif maxValue == theoryB:
            resultList.append(1)
        else:
            resultList.append(2)
    return resultList

def getTestingData():
    testingData = []
    actualPredictions = []
    for i in range(200):
        x, y = blackbox31.ask()
        print(x, y)
        # deep copy the list
        testingData.append(x[:])
        actualPredictions.append(y)
    return testingData, actualPredictions

if __name__ == "__main__":
    testingData, actualPrediction = getTestingData()
    accuracyList = []
    accuracy = 0
    for i in range(1, 1001):
        x, y = blackbox31.ask()
        incrementalLearning(x, y, i)
        print(theoryList)
    myPredictions = testPredictions(testingData, actualPrediction)
    print("Accuracy: ", calculateAccuracy(myPredictions, actualPrediction))
