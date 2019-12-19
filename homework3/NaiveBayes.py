import math
import sys
import os
from Blackbox31 import blackbox31
from Blackbox32 import blackbox32

numberOfTestingSamples = 200
mean, variance, labelCount = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]
accuracyListForGraph = []

#Calculates Gaussian (Normal) probability based on the input value, mean and variance of the corresponding target class
def calculateGaussianProbability(x, index, targetClass):
    numerator = math.exp(-math.pow((float(x)) - (float(mean[targetClass][index])), 2) / (2 * variance[targetClass][index]))
    denominator = math.sqrt(2 * math.pi * variance[targetClass][index])
    return numerator / denominator

def findPredictionsOnTestData(testDataTotal):
    tempPredictions = []
    for testData in testDataTotal:

        #Calculating probabilities for class 0
        probabilityX0 = calculateGaussianProbability(testData[0], 0, 0)
        probabilityY0 = calculateGaussianProbability(testData[1], 1, 0)
        probabilityZ0 = calculateGaussianProbability(testData[2], 2, 0)
        probabilityClass0 = probabilityX0 * probabilityY0 * probabilityZ0 * (labelCount[0] / sum(labelCount))

        #calculating probabilities for class 1
        probabilityX1 = calculateGaussianProbability(testData[0], 0, 1)
        probabilityY1 = calculateGaussianProbability(testData[1], 1, 1)
        probabilityZ1 = calculateGaussianProbability(testData[2], 2, 1)
        probabilityClass1 = probabilityX1 * probabilityY1 * probabilityZ1 * (labelCount[1] / sum(labelCount))

        # calculating probabilities for class 2
        probabilityX2 = calculateGaussianProbability(testData[0], 0, 2)
        probabilityY2 = calculateGaussianProbability(testData[1], 1, 2)
        probabilityZ2 = calculateGaussianProbability(testData[2], 2, 2)
        probabilityClass2 = probabilityX2 * probabilityY2 * probabilityZ2 * (labelCount[2] / sum(labelCount))

        #finds maximum probability of the three and then adds it to the prediction
        maxProbability = max(max(probabilityClass0, probabilityClass1), probabilityClass2)
        if maxProbability == probabilityClass0:
            tempPredictions.append(0)
        elif maxProbability == probabilityClass1:
            tempPredictions.append(1)
        else:
            tempPredictions.append(2)
    return tempPredictions

#maintains a running update of the number of samples seen in that particular class so far
def updateLabelCount(target):
    labelCount[target] += 1

#maintains a running update of mean
def updateMean(data, target):
    for i in range(3):
        mean[target][i] = mean[target][i] + ((data[i] - mean[target][i]) / labelCount[target])

#maintains a running update of variance
def updateVariance(data, target):
    for i in range(3):
        variance[target][i] = (((labelCount[target] - 2) / (labelCount[target] - 1)) * variance[target][i]) + (math.pow((data[i] - mean[target][i]), 2) / labelCount[target])

#maintaining thee tables - count, mean and variance which needs to be updated each time we get a new sample
def updateTables(data, target, index):
    updateLabelCount(target)
    if labelCount[target] == 1:
        for i in range(3):
            variance[target][i] = (math.pow((data[i] - mean[target][i]), 2) / labelCount[target])
    if labelCount[target] > 1:
        updateVariance(data, target)
    updateMean(data, target)

#function to query the blackbox and get the testing data
def getTestingData(blackBoxName):
    testingData = []
    actualPredictions = []
    for i in range(numberOfTestingSamples):
        if blackBoxName == 'blackbox31':
            x, y = blackbox31.ask()
        else:
            x, y = blackbox32.ask()
        #print(x, y)
        testingData.append(x[:])
        actualPredictions.append(y)
    return testingData, actualPredictions

#will calculate accuracy based on the number of mismatches and the 200 test data that we set aside
def calculateAccuracy(myPrediction, actualPrediction):
    mismatch = 0
    for myPred, actualPred in zip(myPrediction, actualPrediction):
        if myPred != actualPred:
            mismatch += 1
    return (numberOfTestingSamples - mismatch) / numberOfTestingSamples

#function to write results to a .txt file
def writeResultsToFile(accuracyList, blackBoxName):
    fileName = 'results_' + blackBoxName + '.txt'
    fileHandle = open(fileName, 'w')
    count = 0
    for i in range(10, 1001, 10):
        fileHandle.write(str(i) + ", " + str(accuracyList[count]) + "\n")
        count += 1
    fileHandle.close()

if __name__ == "__main__":
    blackBoxName = sys.argv[-1]
    blackBoxName = os.path.basename(blackBoxName)
    testingData, actualPrediction = getTestingData(blackBoxName)
    accuracyList = []
    accuracy = 0
    for i in range(1, 1001):
        if blackBoxName == 'blackbox31':
            x, y = blackbox31.ask()
        else:
            x, y = blackbox32.ask()
        updateTables(x, y, i)
        if i % 10 == 0:
            myPrediction = findPredictionsOnTestData(testingData)
            accuracyListForGraph.append(calculateAccuracy(myPrediction, actualPrediction))
    #print("accuracy", accuracyListForGraph)

    writeResultsToFile(accuracyListForGraph, blackBoxName)

