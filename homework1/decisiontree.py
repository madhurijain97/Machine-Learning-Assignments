import sys
from math import log
import csv
import numpy as np
import os

#if all the decisions in the set belong to the same class, then this value will be returned
ifSetPure = 0

#the predictions of the test data are stored in this list too along with being written to the predctions file
testedPredictions = []


'''the structure of the tree
left- left subtree
right - right subtree
value - the value of the node at which it is split
entropy - entropy if the dataset is split at that value
leftDecision - if the left children do not need to be split further, then add a decision to the left decision of the node
rightDecision - if the right children do not have to be split further, then add a decision to the right decision of the node
'''
class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None
        self.entropy = None
        self.leftDecision = None
        self.rightDecision = None
        self.columnNumber = None

'''this calculates the entropy of the entire dataset
entropy = (-plogp - nlogn)'''
def calculateIndividualEntropy(positive, negative, totalSamples):
    positiveProbability = float(positive) / float(totalSamples)
    negativeProbability = float(negative) / float(totalSamples)
    individualEntropy = 0
    if positive != 0 and totalSamples != 0:
        individualEntropy -= ((positiveProbability * log(positiveProbability) / log(2)))
    if negative != 0 and totalSamples != 0:
        individualEntropy -= ((negativeProbability * log(negativeProbability) / log(2)))
    return individualEntropy

#calculates the number of samples have decision as 1(positive) and 0(negative)
def calculatePositiveNegative(dataset):
    positive, negative = 0, 0
    for row in dataset:
        if (row[-1] == '1'):
            positive += 1
        else:
            negative += 1
    return positive, negative

'''Calculates the entropy if a there are left and right children of a node
entropy = (leftSample/totalSample)(-leftp*log(leftp)-(leftn*log(leftn))) + (rightSample/totalSample)(-rightp*log(rightp)-(rightn*log(rightn))) 
'''
def getEntropy(groups):
    totalSample = 0
    leftRightChildEntropies = []
    # get the total samples in the two dataset
    for group in groups:
        totalSample += len(group)
        totalSample = float(totalSample)
    for group in groups:
        individualGroupLength = len(group)
        if individualGroupLength == 0:
            continue
        entropy = 0
        positive, negative = calculatePositiveNegative(group)
        leftRightChildEntropies.append(
            (float(len(group)) / float(totalSample)) * calculateIndividualEntropy(positive, negative,
                                                                                  individualGroupLength))
    return sum(leftRightChildEntropies)

'''We don't want to split on the same nodes again and again
Hence getting the unique values and then splitting
This reduces the number of redundant computations'''
def getUniqueValues(dataset, columnNumber):
    duplicateValuedList = []
    for row in dataset:
        duplicateValuedList.append(int(row[columnNumber]))
    uniqueValuesList = np.unique(duplicateValuedList)
    return uniqueValuesList

'''
This function iterates through the entire dataset
All the values that are less than or equal to nodeValue are added to the leftChildren and
all the values that are greater than nodeValue are added to the rightChildren
'''
def getLeftAndRightChildren(dataset, value, columnNumber):
    left, right = [], []
    for row in dataset:
        if int(row[columnNumber]) <= int(value):
            left.append(row)
        else:
            right.append(row)
    return left, right

'''
Iterates through every value, gets the left and right children, calculates entropy
The value with the least entropy is chosen as the splitNode
'''
def calculateSplitPoint(dataset, entropyOfParent):
    splitColumn = float('inf')
    splitValue = float('inf')
    minimumEntropy = float('inf')
    leftRightChildrenGroups = None
    positive, negative = calculatePositiveNegative(dataset)
    datasetEntropy = calculateIndividualEntropy(positive, negative, positive+negative)
    # running for the first two columns
    for columnNumber in range(2):
        uniqueList = getUniqueValues(dataset, columnNumber)
        for i in range(len(uniqueList)-1):
            number1 = uniqueList[i]
            number2 = uniqueList[i+1]
            number = (number1+number2)/2
            groups = getLeftAndRightChildren(dataset, number, columnNumber)
            entropy = getEntropy(groups)
            if entropy < minimumEntropy and entropy < datasetEntropy:
                minimumEntropy = entropy
                splitColumn = columnNumber
                splitValue = number
                leftRightChildrenGroups = groups
    #if no node could be found with lesser entropy than the entropy of the entire dataset, then return 2(invalid) as the columnNumber
    if splitValue == float('inf'):
        return {'columnNumber': 2}
    #return the details needed for a node
    return {'columnNumber': splitColumn, 'splitValue': splitValue, 'groups': leftRightChildrenGroups,
            'entropy': minimumEntropy}

'''Calculates the number of positives and negatives in the dataset and returns the decision that has the maximum count'''
def assignLeafNode(group):
    positive, negative = 0, 0
    for row in group:
        if row[-1] == '1':
            positive += 1
        else:
            negative += 1
    if (positive > negative):
        return 1
    return 0

'''
decides if a node has to be split further or a decision can be assigned'''
def decideLeafOrSplit(treeNode, leftChildrenList, rightChildrenList, maxDepth, depth):

    #if the current depth is greater than the maximum depth or entropy < 0.01, assign decision nodes and return
    if (depth >= maxDepth or treeNode.entropy < 0.01):
        treeNode.leftDecision = assignLeafNode(leftChildrenList)
        treeNode.rightDecision = assignLeafNode(rightChildrenList)
        return

    #calculate entropy of the leftDataset
    positive, negative = calculatePositiveNegative(leftChildrenList)
    datasetEntropyLeft = calculateIndividualEntropy(positive, negative, positive+negative)

    #calculate entropy of the right dataset
    positive, negative = calculatePositiveNegative(rightChildrenList)
    datasetEntropyRight = calculateIndividualEntropy(positive, negative, positive+negative)

    #if the left entropy < 0.01, no need to split further and a decision node can be assigned
    if datasetEntropyLeft < 0.01:
        treeNode.leftDecision = assignLeafNode(leftChildrenList)
    else:
        #else, get the split point
        newTreeNodeDict = calculateSplitPoint(leftChildrenList, treeNode.entropy)
        #if no new node could be found, assign a decision to the current node
        if(newTreeNodeDict['columnNumber'] == 2):
            treeNode.leftDecision = assignLeafNode(leftChildrenList)
        else:
            #else, create a new node and call this function recursively
            newTreeNode = Tree()
            treeNode.left = newTreeNode
            newTreeNode.value = newTreeNodeDict['splitValue']
            newTreeNode.columnNumber = newTreeNodeDict['columnNumber']
            newTreeNode.entropy = newTreeNodeDict['entropy']
            toBeLeftChildren, toBeRightChildren = None, None
            if(newTreeNodeDict['groups'] != None):
                toBeLeftChildren = newTreeNodeDict['groups'][0]
                toBeRightChildren = newTreeNodeDict['groups'][1]
            decideLeafOrSplit(newTreeNode, toBeLeftChildren, toBeRightChildren, maxDepth, depth + 1)

    # if the right entropy < 0.01, no need to split further and a decision node can be assigned
    if datasetEntropyRight < 0.01:
        treeNode.rightDecision = assignLeafNode(rightChildrenList)
    else:
        # else, get the split point
        newTreeNodeDict = calculateSplitPoint(rightChildrenList, treeNode.entropy)
        # if no new node could be found, assign a decision to the current node
        if (newTreeNodeDict['columnNumber'] == 2):
            treeNode.rightDecision = assignLeafNode(rightChildrenList)
        else:
            # else, create a new node and call this function recursively
            newTreeNode = Tree()
            treeNode.right = newTreeNode
            newTreeNode.value = newTreeNodeDict['splitValue']
            newTreeNode.columnNumber = newTreeNodeDict['columnNumber']
            newTreeNode.entropy = newTreeNodeDict['entropy']
            toBeLeftChildren, toBeRightChildren = None, None
            if (newTreeNodeDict['groups'] != None):
                toBeLeftChildren = newTreeNodeDict['groups'][0]
                toBeRightChildren = newTreeNodeDict['groups'][1]
            decideLeafOrSplit(newTreeNode, toBeLeftChildren, toBeRightChildren, maxDepth, depth + 1)

#this function is called while classifying the test data
def classify(root, x, y):
    #if root is None, means that the entire dataset is pure. Hence returning that value
    if root == None or root is None:
        return ifSetPure
    #do down the tree as per the conditions that are being satisfied
    if root is not None:
        while (True):
            if (root.columnNumber == 0):
                if int(x) <= int(root.value):
                    if root.leftDecision != None:
                        return root.leftDecision
                    root = root.left
                else:
                    if root.rightDecision != None:
                        return root.rightDecision
                    root = root.right
            elif(root.columnNumber == 1):
                if int(y) <= int(root.value):
                    if root.leftDecision != None:
                        return root.leftDecision
                    root = root.left
                else:
                    if root.rightDecision != None:
                        return root.rightDecision
                    root = root.right
    else:
        return -1

#gets the root value of the tree and recursively constructs the tree using decideLeafOrSplit function
def buildDecisionTree(dataset, maxDepth, minimumEntropy):
    if(minimumEntropy == 0):
        return None
    rootDict = calculateSplitPoint(dataset, minimumEntropy)
    root = Tree()
    root.value = rootDict['splitValue']
    root.columnNumber = rootDict['columnNumber']
    root.entropy = rootDict['entropy']
    toBeLeftChildren, toBeRightChildren = None, None
    if rootDict['groups'] != None:
        toBeLeftChildren = rootDict['groups'][0]
        toBeRightChildren = rootDict['groups'][1]
    decideLeafOrSplit(root, toBeLeftChildren, toBeRightChildren, maxDepth, 1)
    return root

#open the file that contains the training data and store it in a list
def readingCsvFile(fileName):
    fileHandle = open(fileName)
    with fileHandle as csvFile:
        csvReader = csv.reader(csvFile)
        globalDataset = list(csvReader)
        count = 0
        for row in globalDataset:
            ifSetPure = int(row[-1])
            count += 1
            if(count > 0):
                break

    positive, negative = calculatePositiveNegative(globalDataset)
    fileHandle.close()
    return globalDataset, calculateIndividualEntropy(positive, negative, positive+negative)

#Open the file that contains the test data and write your predictions to a file
def testPredictionsOnDecisionTree(root, testingFile):
    testedPredictions.clear()
    fileHandle1 = open(testingFile)
    basePathName = os.path.basename(testingFile)
    wordsList = basePathName.split('_')
    blackBoxNumber = wordsList[0]
    myPredictionsFileName = blackBoxNumber + "_" + "predictions.csv"
    #print("predictionsFilename is:", myPredictionsFileName)
    fileHandle2 = open(myPredictionsFileName, 'w')
    with fileHandle1 as csvFile:
        csvReader = csv.reader(csvFile)
        testingDataSet = list(csvReader)
        for data in testingDataSet:
            decision = classify(root, int(data[0]), int(data[1]))
            fileHandle2.write(str(decision) + '\n')
            testedPredictions.append(decision)
    fileHandle1.close()
    fileHandle2.close()


'''gets the training and testing file names and calls buildDecisionTree to start the process of building a tree,
Calls testPredictionsOnDecisionTrees to start classifying the data
'''
if __name__ == "__main__":
    trainingDataPath, testingDataPath = sys.argv[1], sys.argv[2]
    globalDataset, minimumEntropy = readingCsvFile(trainingDataPath)
    finalRoot = None
    if minimumEntropy != 0:
        finalRoot = buildDecisionTree(globalDataset, 9, minimumEntropy)

    testPredictionsOnDecisionTree(finalRoot, testingDataPath)


