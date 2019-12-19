import sys
import utils
import csv
import math
import numpy as numpy
import os

errorForPlottingGraphList, accuracyScoreForGraph = [], []
numberOfSamples, numOfEpochs = 0, 200

#Will take derivative of relu
def inplace_relu_derivative(someOutput):
    someOutput[someOutput > 0] = 1
    someOutput[someOutput <= 0] = 0
    return someOutput

def backPropagation(input, OOutput, truePrediction, binarizedTruePrediction, h1Output, h2Output, numberOfNeuronsInLayers, classes, h1In, h2In, OIn, optimizer):
    #derivative of cross entropy wrt weights3
    delta_temp_output = numpy.subtract(OOutput, binarizedTruePrediction)
    delta3 = numpy.dot(h2Output.transpose(), delta_temp_output)

    #derivative of cross entropy wrt weights2
    tempArrayForh2 = numpy.copy(h2In)
    inplaceReluDerivativeOfh2 = inplace_relu_derivative(tempArrayForh2)
    delta_temp_second_hidden_layer = numpy.dot(delta_temp_output, weights[2].transpose())
    delta2 = h1Output.transpose().dot(numpy.multiply(delta_temp_second_hidden_layer, inplaceReluDerivativeOfh2))

    #derivative of cross entropy wrt weights1
    tempArrayForh1 = numpy.copy(h1In)
    inplaceReluDerivativeOfh1 = inplace_relu_derivative(tempArrayForh1)
    delta_temp_first_hidden_layer = delta_temp_second_hidden_layer.dot(weights[1].transpose())
    delta1 = input[:, :-1].transpose().dot(numpy.multiply(delta_temp_first_hidden_layer, inplaceReluDerivativeOfh1))

    #derivative of cross entropy wrt bias3
    biasDelta3 = numpy.copy(delta_temp_output)

    #derivative of cross entropy wrt bias2
    bias_delta_temp_second_hidden_layer = numpy.dot(delta_temp_output, numpy.repeat(numpy.array([bias[2]]), repeats = [numberOfNeuronsInLayers[2]], axis = 0).transpose())
    biasDelta2 = numpy.multiply(bias_delta_temp_second_hidden_layer, inplaceReluDerivativeOfh2)

    #derivative of cross entropy wrt bias1
    bias_delta_temp_first_hidden_layer = numpy.dot(bias_delta_temp_second_hidden_layer, numpy.repeat(numpy.array([bias[1]]), repeats = [numberOfNeuronsInLayers[1]], axis = 0).transpose())
    biasDelta1 = numpy.multiply(bias_delta_temp_first_hidden_layer, inplaceReluDerivativeOfh1)

    bias3Mean = biasDelta3.mean(axis = 0)
    bias2Mean = biasDelta2.mean(axis = 0)
    bias1Mean = biasDelta1.mean(axis = 0)

    grads = [delta1, delta2, delta3, bias1Mean, bias2Mean, bias3Mean]
    optimizer.update_params(grads)
    weights[2] = optimizer.params[2]
    weights[1] = optimizer.params[1]
    weights[0] = optimizer.params[0]
    bias[2] = optimizer.params[-1]
    bias[1] = optimizer.params[-2]
    bias[0] = optimizer.params[-3]

def forwardPropagation(input, weights, bias, originalOutput, binarizedTruePrediction, prediction, numberOfSamples, numberOfNeuronsInLayers, classes, optimizer):
    #computing output of first hidden layer
    h1In = numpy.dot(input[:, :3], weights[0]) + numpy.repeat(numpy.array([bias[0]]), repeats = [numberOfSamples], axis = 0)
    h1Output = utils.relu(h1In)

    #computing output of second hidden layer
    h2In = numpy.dot(h1Output, weights[1]) + numpy.repeat(numpy.array([bias[1]]), repeats = [numberOfSamples], axis = 0)
    h2Output = utils.relu(h2In)

    #computing output of the output layer
    OIn = numpy.dot(h2Output, weights[2]) + numpy.repeat(numpy.array([bias[2]]), repeats = [numberOfSamples], axis = 0)
    OOutput = utils.softmax(OIn)

    myPredictedValueListAsIntegers = numpy.argmax(OOutput, axis=1)
    # Computing overall error only for plotting graph
    if prediction == False:
        errorForGraph = utils.log_loss(binarizedTruePrediction, OOutput[:] + 0.00001)
        errorForPlottingGraphList.append(errorForGraph)
        accuracyScoreForGraph.append(utils.accuracy_score(originalOutput, myPredictedValueListAsIntegers))
        backPropagation(input, OOutput, originalOutput, binarizedTruePrediction, h1Output, h2Output, numberOfNeuronsInLayers, classes, h1In, h2In, OIn, optimizer)
    else:
        return myPredictedValueListAsIntegers

#initialize weights and bias
def initializeWeights(numberOfNeuronsInLayers):
    weights, bias = [], []

    for i in range(1, 4):
        weights.append(numpy.random.randn(numberOfNeuronsInLayers[i - 1], numberOfNeuronsInLayers[i]) * math.sqrt(
            1 / numberOfNeuronsInLayers[i - 1]))
        bias.append(numpy.random.randn(numberOfNeuronsInLayers[i]) * math.sqrt(1 / numberOfNeuronsInLayers[i - 1]))
    return weights, bias


def readingCsvFile(fileName):
    fileHandle = open(fileName)
    with fileHandle as csvFile:
        csvReader = csv.reader(csvFile)
        dataset = list(csvReader)
    fileHandle.close()
    return dataset


def makePredictionOnTestingSet(testingFile, weights, bias, numberOfNeuronsInLayers, classes, optimizer):
    fileHandle1 = open(testingFile)
    basePathName = os.path.basename(testingFile)
    wordsList = basePathName.split('_')
    blackBoxNumber = wordsList[0]
    myPredictionsFileName = blackBoxNumber + "_" + "predictions.csv"
    fileHandle2 = open(myPredictionsFileName, 'w')

    with fileHandle1 as csvFile:
        csvReader = csv.reader(csvFile)
        testingDataSetTemp = list(csvReader)
        testingDataSet = numpy.array([[int(i) for i in row] for row in testingDataSetTemp])
        numberOfSamples = len(testingDataSet)
        testedPredictions = forwardPropagation(testingDataSet, weights, bias, None, None, True, numberOfSamples, numberOfNeuronsInLayers, classes, optimizer)
        for value in testedPredictions:
            fileHandle2.write(str(value) + '\n')
    fileHandle1.close()
    fileHandle2.close()

    return testedPredictions

def makePrediction(trainingFile, testingFile, weights, bias, numberOfNeuronsInLayers, classes, blackBoxNumber, optimizer):
    testedPredictions = makePredictionOnTestingSet(testingFile, weights, bias, numberOfNeuronsInLayers, classes, optimizer)

if __name__ == "__main__":
    trainingDataPath, testingDataPath = sys.argv[1], sys.argv[2]
    #trainingDataPath, testingDataPath = 'Data/blackbox22_train.csv', 'Data/blackbox22_test.csv'

    dataset = readingCsvFile(trainingDataPath)
    numberOfSamples = len(dataset)
    globalDataset = [[int(i) for i in row] for row in dataset]
    globalDataset = numpy.array(globalDataset)

    basePathName = os.path.basename(trainingDataPath)
    wordsList = basePathName.split('_')
    blackBoxNumber = wordsList[0]

    if blackBoxNumber == "blackbox21":
        classes = [0, 1, 2]
        numberOfNeuronsInLayers = [3, 10, 20, 3]
    elif blackBoxNumber == "blackbox22":
        classes = [0, 1, 2, 3, 4, 5]
        numberOfNeuronsInLayers = [3, 10, 20, 6]
    else:
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        numberOfNeuronsInLayers = [3, 10, 20, 8]

    # get a list of original outputs --> originalOutput = [[0],[1], [0], [2], [7]] where each value corresponds to output of that sample
    originalOutput = globalDataset[:, 3]
    binarizedTruePrediction = utils.label_binarize(originalOutput, classes)
    numpy.random.seed(1)

    weights, bias = initializeWeights(numberOfNeuronsInLayers)
    optimizer = utils.AdamOptimizer(weights + bias)

    for epoch in range(numOfEpochs):
        start = 0
        while start < len(globalDataset):
            if start + 5000 > len(globalDataset):
                numberOfSamples = len(globalDataset) - start
            else:
                numberOfSamples = 5000
            forwardPropagation(globalDataset[start : start + 5000], weights, bias, originalOutput[start : start + 5000], binarizedTruePrediction[start : start + 5000], False, numberOfSamples, numberOfNeuronsInLayers, classes, optimizer)
            start += 5000

    makePrediction(trainingDataPath, testingDataPath, weights, bias, numberOfNeuronsInLayers, classes, blackBoxNumber, optimizer)

