import sys
import os
import copy

gamma, epsilon, initialUtility = 0.9, 0.1, 0.0
reward, penalty = 99, -101

#used to fill the utility values depending on the direction
def fillUtility(grid, i, j, up, down, left, right):
    upProb, downProb, leftProb, rightProb = 0.1, 0.1, 0.1, 0.1
    if up:
        upProb = 0.7
    elif down:
        downProb = 0.7
    elif right:
        rightProb = 0.7
    elif left:
        leftProb = 0.7

    stateValue = 0
    # Moving Up
    if i - 1 >= 0:
        stateValue += upProb * grid[i - 1][j]
    else:
        stateValue += upProb * grid[i][j]

    # Moving left
    if j - 1 >= 0:
        stateValue += leftProb * grid[i][j - 1]
    else:
        stateValue += leftProb * grid[i][j]

    # Moving Down
    if i + 1 < gridSize:
        stateValue += downProb * grid[i + 1][j]
    else:
        stateValue += downProb * grid[i][j]

    # Moving Right
    if j + 1 < gridSize:
        stateValue += rightProb * grid[i][j + 1]
    else:
        stateValue += rightProb * grid[i][j]

    return stateValue

#if the difference between the two computed grids of utility values is less than the allowed error, then stop computing the utility values
def compareForStoppingCondition(grid, newGrid):
    numOfMatches = 0
    for i in range(gridSize):
        for j in range(gridSize):
            if (abs(grid[i][j] - newGrid[i][j]) < allowedError):
                numOfMatches += 1

    if (numOfMatches == gridSize * gridSize):
        return True
    return False

#this function computes the appropriate utility for a block using the value iteration formula
def computeUtilities(grid, newGrid, initialReward, directionList,destY, destX):
    shouldStop = False
    count = 0
    finalDirectionList = []
    while(shouldStop == False):
        count += 1
        for i in range(gridSize):
            for j in range(gridSize):
                if i == destX and j == destY:
                    continue

                tempUp = fillUtility(grid, i, j, True, False, False, False)
                tempDown = fillUtility(grid, i, j, False, True, False, False)
                tempRight = fillUtility(grid, i, j, False, False, False, True)
                tempLeft = fillUtility(grid, i, j, False, False, True, False)

                temp_grid_val = tempUp
                if tempDown > temp_grid_val:
                    temp_grid_val = tempDown

                if tempRight > temp_grid_val:
                    temp_grid_val = tempRight

                if tempLeft > temp_grid_val:
                    temp_grid_val = tempLeft

                newGrid[i][j] = gamma * temp_grid_val + initialReward[i][j]
                # Breaking the tie as north, south, east, west
                if(initialReward[i][j] != penalty and initialReward[i][j] != reward):
                    if newGrid[i][j] == gamma * tempUp + initialReward[i][j]:
                        directionList[i][j] = "^"
                    elif newGrid[i][j] == gamma * tempDown + initialReward[i][j]:
                        directionList[i][j] = "v"
                    elif newGrid[i][j] == gamma * tempRight + initialReward[i][j]:
                        directionList[i][j] = ">"
                    elif newGrid[i][j] == gamma * tempLeft + initialReward[i][j]:
                        directionList[i][j] = "<"
        shouldStop = compareForStoppingCondition(grid, newGrid)
        if shouldStop == False:
            grid = copy.deepcopy(newGrid)
            finalDirectionList = copy.deepcopy(directionList)
    return directionList, grid

#this function is just used for writing the results to a file
def writeResultsToFile(finalDirectionList, outputFileName):
    fileName = outputFileName
    fileHandle = open(fileName, 'w')
    for i in range(gridSize):
        for j in range(gridSize):
            fileHandle.write(str(finalDirectionList[i][j]))
        if i != gridSize - 1:
            fileHandle.write("\n")
    fileHandle.close()

if __name__ == "__main__":
    inputTextFileName, outputFileName = sys.argv[1], sys.argv[2]
    fileHandle = open(inputTextFileName);

    gridSize = int(fileHandle.readline())
    grid = [[0 for i in range(gridSize)] for j in range(gridSize)]
    newGrid = [[0 for i in range(gridSize)] for j in range(gridSize)]
    initialReward = [[-1 for i in range(gridSize)] for j in range(gridSize)]
    directionList = [["a" for i in range(gridSize)] for j in range(gridSize)]

    numOfObstacles = int(fileHandle.readline())
    for i in range(numOfObstacles):
        y, x = fileHandle.readline().split(",")
        grid[int(x)][int(y)] = 0

        initialReward[int(x)][int(y)] = penalty
        directionList[int(x)][int(y)] = "x"

    destY, destX = fileHandle.readline().split(",")
    grid[int(destX)][int(destY)] = reward
    newGrid[int(destX)][int(destY)] = reward
    initialReward[int(destX)][int(destY)] = reward
    directionList[int(destX)][int(destY)] = "."

    allowedError = (epsilon * (1 - gamma)) / gamma
    finalDirectionList, grid = computeUtilities(grid, newGrid, initialReward, directionList, int(destY), int(destX))

    writeResultsToFile(finalDirectionList, outputFileName)

