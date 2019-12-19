import numpy as np
import math
from Blackbox31 import blackbox31

numOfClasses, numberOfSamples = 3, 1000
feature1, feature2, feature3 = [], [], []
evidence = 1 / numberOfSamples
theoryList = [1/3, 1/3, 1/3]
labelList = [0, 0, 0]

def calculateVariance(mean1, mean2, mean3):
    varianceList1= [value - mean1 for value in feature1]
    variance1 = (sum(map(lambda value: value * value, varianceList1))) / len(feature1)
    varianceList2 = [value - mean2 for value in feature2]
    variance2 = (sum(map(lambda value: value * value, varianceList2))) / len(feature2)
    varianceList3 = [value - mean3 for value in feature3]
    variance3 = (sum(map(lambda value: value * value, varianceList3))) / len(feature3)
    return variance1, variance2, variance3

def calculateMean():
    mean1 = sum(feature1) / len(feature1)
    mean2 = sum(feature2) / len(feature2)
    mean3 = sum(feature3) / len(feature3)
    return mean1, mean2, mean3

def appendFeatures(inputList):
    feature1.append(inputList[0])
    feature2.append(inputList[1])
    feature3.append(inputList[2])

def preprocessing(inputList):
    appendFeatures(inputList)
    mean1, mean2, mean3 = calculateMean()
    meanList = [mean1, mean2, mean3]
    variance1, variance2, variance3 = calculateVariance(mean1, mean2, mean3)
    varianceList = [variance1, variance2, variance3]
    return meanList, varianceList

def calculateProbability(data, target, meanList, varianceList):
    finalProbabilityList = []
    for label in range(len(data)):
        individualProbabilityList = []
        for dataValue in range(len(data)):
            numerator = math.exp(-math.pow((float(data[dataValue]) - float(meanList[label])), 2) / (2 * varianceList[label]))
            denominator = math.sqrt(2 * math.pi * varianceList[label])
            probability = numerator / denominator
            individualProbabilityList.append(probability)
        finalProbabilityList.append(np.prod(np.array(individualProbabilityList)) * labelList[target] / 1000)
    return finalProbabilityList

def calculateProbabilityForFirstSample():
    pass

def incrementalLearningForOtherSamples(data, target, meanList, varianceList):
    #global theoryList
    print("theoryList", theoryList)
    probabilityList = calculateProbability(data, target, meanList, varianceList)
    for i in range(numOfClasses):
        theoryList[i] = max(theoryList[i], (theoryList[i] * probabilityList[i]))

def incrementalLearning(data, target, index, meanList, varianceList):
    if index == 1:
        calculateProbabilityForFirstSample()
    else:
        incrementalLearningForOtherSamples(data, target, meanList, varianceList)

def calculateAccuracy(myPredictions, actualPredictions):
    count = 0
    for myPred, actualPred in zip(myPredictions, actualPredictions):
        if myPred != actualPred:
            print("Mismatch: ", myPred, actualPred)
            count += 1
    return (numberOfSamples - count) / numberOfSamples

def testPredictions(testingData, meanList, varianceList):
    global labelList
    resultList = []
    labelList = [1000, 1000, 1000]
    for inputList in testingData:
        probabilityList = calculateProbability(inputList, 0, meanList, varianceList)
        theory, maxValue = 0, 0
        for i in range(numOfClasses):
            probability = theoryList[i] * probabilityList[i]
            if probability > theory:
                theory = probability
                maxValue = i
        resultList.append(maxValue)
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
        meanList, varianceList = preprocessing(x)
        labelList[y] += 1
        if i > 10:
            incrementalLearning(x, y, i, meanList, varianceList)
    myPredictions = testPredictions(testingData, meanList, varianceList)
    print("Accuracy: ", calculateAccuracy(myPredictions, actualPrediction))
    print(myPredictions)










