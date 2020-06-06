import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial import distance
np.seterr(divide='ignore', invalid='ignore')

class Kmeans:
    def __init__(self, points, numOfCentroids):
        self.points = points
        self.pointsId = np.zeros(points.shape[0]).astype(int)
        self.numCents = numOfCentroids
        self.n = points.shape[0]

        # init centroids randomly
        centroidsIndex = random.sample(range(0, self.n), numOfCentroids)
        self.centroids = points[centroidsIndex]

    def kmeans(self):
        # iterate until convergence
        newCentroids = np.empty(1)
        oldCentroids = self.centroids
        iterations = 0
        while(np.all(newCentroids != oldCentroids)):
            oldCentroids = newCentroids
            self.kmeans_2()
            newCentroids = self.centroids
            iterations += 1
        print('Num of iterations:', iterations)

    def kmeans_2(self):
        for i in range(self.n):
            # calculate distance between point and centroids
            tempPoint = self.points[i].reshape((1,2))
            dists = distance.cdist(self.centroids,tempPoint)
            # assign point to nearest centroid
            index = np.argmin(dists)
            self.pointsId[i] = index

        # update centroids
        sumPoints = np.zeros((self.numCents, 2))
        countPoints = np.zeros((self.numCents,1))
        for i in range(self.n):
            idx = self.pointsId[i]
            sumPoints[idx] += self.points[i]
            countPoints[idx] += 1

        # average out points
        sumPoints = np.nan_to_num(sumPoints/countPoints)
        # update centroids
        self.centroids = sumPoints
    
    def accuracy(self, y):
        correct = 0
        for i in range(self.n):
            if (y[i] == self.pointsId[i]):
                correct += 1
        print(float(correct/self.n))

    def plot_points(self):
        plt.scatter(self.points[:,0], self.points[:,1], color='blue')
        plt.scatter(self.centroids[:,0], self.centroids[:,1], color='red')
        plt.show()

    def calc_variance(self):
        sumDist = 0
        # iterate through points and sum up distance to respective centroids
        for i in range(self.n):
            index = self.pointsId[i]
            centroidPos = self.centroids[index].reshape((1,2))
            tempPoint = self.points[i].reshape((1,2))
            dist = distance.cdist(centroidPos, tempPoint)
            # print(dist)
            sumDist += dist
        print(sumDist)
        
# Create dataset with 3 random cluster centers and 1000 datapoints
x, y = make_blobs(n_samples = 1000, centers = 3, n_features=2, shuffle=True, random_state=31)
        
k = Kmeans(x, 3)
k.kmeans()
# k.accuracy(y)
k.plot_points()
k.calc_variance()

