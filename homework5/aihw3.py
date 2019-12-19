# homework 3
import copy, math
import numpy as np

fileOpen = open('test2.txt', 'r')
fileOut = open('output.txt', 'w')
obstacles, startLocation, endLocation, finalAnswer = [], [], [], []


# start_time = time.time()

def turnLeft(direction):
    if direction == "up":
        return "left"
    elif direction == "left":
        return "down"
    elif direction == "down":
        return "right"
    else:
        return "up"


def fillUtility(matrix, i, j, up, down, left, right):
    upProb, downProb, leftProb, rightProb = 0.1, 0.1, 0.1, 0.1
    if up:
        upProb = 0.7
        tempMove = "up"
    elif down:
        downProb = 0.7
        tempMove = "down"
    elif right:
        rightProb = 0.7
        tempMove = "right"
    else:
        leftProb = 0.7
        tempMove = "left"

    temp = 0
    # Moving Up
    if i - 1 >= 0:
        temp += upProb * matrix[i - 1][j]
    else:
        temp += upProb * matrix[i][j]

    # Moving left
    if j - 1 >= 0:
        temp += leftProb * matrix[i][j - 1]
    else:
        temp += leftProb * matrix[i][j]

    # Moving Down
    if i + 1 < len(matrix):
        temp += downProb * matrix[i + 1][j]
    else:
        temp += downProb * matrix[i][j]

    # Moving Right
    if j + 1 < len(matrix):
        temp += rightProb * matrix[i][j + 1]
    else:
        temp += rightProb * matrix[i][j]

    returnTemp = temp
    temp = 0
    temp = 0.9 * returnTemp + initialMatrix[i][j]

    return temp, tempMove


def computeUtilities(matrix, newMatrix):
    count = 0
    while count < (sizeOfGrid * sizeOfGrid):
        count = 0
        for i in range(sizeOfGrid):
            for j in range(sizeOfGrid):
                # print(endX, endY)
                if i == endX and j == endY:
                    count += 1
                    continue
                # Moving Left
                tempLeft, tempMoveLeft = fillUtility(matrix, i, j, False, False, True, False)
                # Moving Right
                tempRight, tempMoveRight = fillUtility(matrix, i, j, False, False, False, True)
                # Moving Down
                tempDown, tempMoveDown = fillUtility(matrix, i, j, False, True, False, False)
                # Moving Up
                tempUp, tempMoveUp = fillUtility(matrix, i, j, True, False, False, False)

                originalVal = matrix[i][j]
                newMatrix[i][j] = max(tempUp, tempDown, tempLeft, tempRight)
                if newMatrix[i][j] == tempLeft:
                    move[i][j] = tempMoveLeft
                elif newMatrix[i][j] == tempRight:
                    move[i][j] = tempMoveRight
                elif newMatrix[i][j] == tempDown:
                    move[i][j] = tempMoveDown
                elif newMatrix[i][j] == tempUp:
                    move[i][j] = tempMoveUp

                if abs(originalVal - newMatrix[i][j]) <= 0.1:
                    count += 1
        matrix = copy.deepcopy(newMatrix)


with fileOpen as inputFile:
    lines = inputFile.readlines()

    sizeOfGrid = int(lines[0])
    numOfCars = int(lines[1])
    numOfObstacles = int(lines[2])
    if numOfObstacles != 0:
        for i in range(3, 3 + numOfObstacles):
            obstacles.append((lines[i].strip()).split(","))
            print("obstacles", obstacles)
    else:
        i = 2
    for j in range(i + 1, i + 1 + numOfCars):
        startLocation.append((lines[j].strip()).split(","))
        print("startLocation", startLocation)

    for i in range(j + 1, j + 1 + numOfCars):
        endLocation.append((lines[i].strip()).split(","))
        print("endLocation: ", endLocation)

    # Creation of matrix
    matrix = [[0 for i in range(sizeOfGrid)] for j in range(sizeOfGrid)]
    initialMatrix = [[-1 for i in range(sizeOfGrid)] for j in range(sizeOfGrid)]
    newMatrix = [[0 for i in range(sizeOfGrid)] for j in range(sizeOfGrid)]
    directionList = [[0 for i in range(sizeOfGrid)] for j in range(sizeOfGrid)]
    for obs in obstacles:
        obsX = int(obs[0])
        obsY = int(obs[1])
        initialMatrix[obsX][obsY] -= 100

    for i in range(numOfCars):
        # print("\nCar number: ", i + 1)
        money = 0
        move = [["null" for r in range(sizeOfGrid)] for c in range(sizeOfGrid)]
        start = startLocation[i]
        startX, startY = int(start[0]), int(start[1])
        end = endLocation[i]
        endX, endY = int(end[0]), int(end[1])
        matrix[endX][endY] += 99
        newMatrix[endX][endY] += 99
        initialMatrix[endX][endY] += 100

        computeUtilities(matrix, newMatrix)
        for i in range(len(newMatrix)):
            print(newMatrix[i])

        for j in range(10):
            # Start iterating the matrix as per the random code
            np.random.seed(j)
            swerve = np.random.random_sample(1000000)
            k = 0
            x, y = startX, startY
            if x == endX and y == endY:
                money += 100
            while x != endX or y != endY:
                direction = move[x][y]
                if swerve[k] > 0.7:
                    if swerve[k] > 0.8:
                        if swerve[k] > 0.9:
                            returnedMove = turnLeft(turnLeft(direction))
                        else:
                            returnedMove = turnLeft(direction)
                    else:
                        returnedMove = turnLeft(turnLeft(turnLeft(direction)))
                    direction = returnedMove
                k += 1
                directionList[x][y] = direction
                if direction == "left" and y - 1 >= 0:
                    y = y - 1
                elif direction == "right" and y + 1 < sizeOfGrid:
                    y = y + 1
                elif direction == "down" and x + 1 < sizeOfGrid:
                    x = x + 1
                elif direction == "up" and x - 1 >= 0:
                    x = x - 1
                money += initialMatrix[x][y]

        initialMatrix[endX][endY] = -1
        matrix = [[0 for i in range(sizeOfGrid)] for j in range(sizeOfGrid)]
        newMatrix = [[0 for i in range(sizeOfGrid)] for j in range(sizeOfGrid)]
        finalAnswer.append(math.floor(money / 10))

    for row in directionList:
        print(directionList)
    # for row in move:
    # print(row)
for value in finalAnswer:
    fileOut.write(str(int(value)) + "\n")
    #print(value)
# print("Time taken = %s" % (time.time() - start_time))