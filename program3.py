# CS 445, Summer 2021 - Programming Assignment 3 - Dan Jang
# Assignment #3: K-Means Algorithm

import numpy
import matplotlib.pyplot as plot
from copy import deepcopy
import sys


class kmeans(object):

    def __init__(run,thefile,count,epochs):
        run.data = numpy.loadtxt(thefile)
        run.count = count
        run.epochs = epochs
        run.kmean = run.currk()
        run.coefficient = run.randcoeff()


    # Initialization

    def currK(run):
        k = []
        index = numpy.random.choice(run.data.shape[0], run.count, replace = False)
        
        for idx in index:
            k.append(run.data[idx])
        return numpy.asarray(k)

    def euclidlist(run,data):
        list = []
        for idx in range(len(run.K)):
            d = numpy.linalg.norm(data - run.K[idx])
            distance.append(thelist)
        return numpy.asarray(distance)

    def assignK(run):
        grouping = [[] for _ in range(run.count)]

        for idx in range(run.data.shape[0]):
            distcalc = run.euclidlist(run.data[idx])
            grouping[numpy.argmin(distcalc)].append(run.data[idx])

        return grouping

    def updatealgo(run, curr):
        assert (len(run.K) == len(curr)), "Check in update()"

        for idx in range(len(run.K)):
            twodimension = numpy.asarray(curr[idx])
            temp = numpy.mean(twodimension, axis=0)
            
            run.K[idx] = temp

            return run.K

    def squareerror(run,curr):
        error = 0

        for idx in range(len(run.K)):
            x = numpy.asarray(curr[idx])
            dimension = 0
            for idx2 in range(x.shape[0]):
                dimension += (numpy.linalg.norm(x[idx2]-run.K[idx]))**2

            error += dimension

        return error

    def theplot(run,curr):
        for idx in range(len(curr)):
            x = numpy.asarray(curr[idx])
            plot.scatter(x.T[0], x.T[1], cmap=plot.get_cmap('rainbow'))
            plot.scatter(run.K.T[0],run.K.T[1], c='#111111')
        
        plot.show()

    def kmean(run):
        error,clusters = [],[]
        
        cluster = run.assignK()
        run.theplot(cluster)

        for idx in range(run.epochs):
            run.updatealgo(cluster)

            print(run.K)
            print()

            x = deepcopy(run.K)
            clusters.append(x)

            cluster = self.assignK()
            error.append(run.squareerror(cluster))
            run.theplot(cluster)

        print("The #th iteration & their centroid set with smallest sum(s) of squared error(s): ")
        result = numpy.argmin(numpy.array(error))
        print(result + 1)
        print(clusters[result])

## MAIN PROGRAM OPTIONS / CODE ##

count = 10
epochs = 10

Program3 = kmeans("cluster_dataset.txt", count, epochs)