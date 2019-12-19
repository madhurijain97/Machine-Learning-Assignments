import numpy as np
import sys
import os
from Blackbox41 import Blackbox41
from Blackbox42 import Blackbox42

numOfClusters = 4

def initializeRandomCentroids(data):
    #picking up the first four data points as the four centroids randomly
    centroids = data[:4]
    return centroids

def findMinimumumDisatnce(data, centroids):
    #calculating distance of all the datapoints with each of the 4 centroids
    totalDistance1 = np.sqrt(np.sum((data - centroids[0]) ** 2, axis=1))
    totalDistance2 = np.sqrt(np.sum((data - centroids[1]) ** 2, axis=1))
    totalDistance3 = np.sqrt(np.sum((data - centroids[2]) ** 2, axis=1))
    totalDistance4 = np.sqrt(np.sum((data - centroids[3]) ** 2, axis=1))

    #combining the 4 distance arrays into one
    totalDistance = np.array([totalDistance1, totalDistance2, totalDistance3, totalDistance4])

    #calculate the minimum of them and assign a cluster to it
    minimumDistance = np.argmin(totalDistance, axis=0)

    return minimumDistance

#calculating new centroids based on the mean of the clusters
def relocateCentroid(data, minimumDistance):
    newCentroids = []
    for k in range(numOfClusters):
        newCentroids.append(data[minimumDistance == k].mean(axis=0))
    return np.asarray(newCentroids)

def writeResultsToFile(minimumDistance, blackBoxName):
    fileName = 'results_' + blackBoxName + '.csv'
    fileHandle = open(fileName, 'w')

    with fileHandle as csvFile:
        for distance in minimumDistance:
            fileHandle.write(str(distance) + '\n')
    fileHandle.close()

#find the clusters and the centroids
#Stopping condition: centroids remain the same in the two consecutive iterations
def performClustering(centroids):
    count = 0
    while(True):
        count += 1
        oldCentroid = centroids
        minimumDistance = findMinimumumDisatnce(data, centroids)
        centroids = relocateCentroid(data, minimumDistance)
        if (np.array_equal(oldCentroid, centroids)):
            return minimumDistance, centroids
    return minimumDistance, centroids

if __name__ == "__main__":
    blackBoxName = sys.argv[-1]
    blackBoxName = os.path.basename(blackBoxName)
    if blackBoxName == 'blackbox41':
        data = Blackbox41.ask(Blackbox41)
    elif blackBoxName == 'blackbox42':
        data = Blackbox42.ask(Blackbox42)
    else:
        print("Invalid blackbox")
        sys.exit()
    centroids = initializeRandomCentroids(data)
    minimumDistance, newCentroids = performClustering(centroids)
    writeResultsToFile(minimumDistance, blackBoxName)

