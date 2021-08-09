# CS 445, Summer 2021 - Programming Assignment 3 - Dan Jang
# Assignment #3: K-Means Algorithm

import numpy
import matplotlib.pyplot as plot
from copy import deepcopy


class kmeans(object):

    def __init__(run,thefile,count,epochs):

        run.data = numpy.loadtxt(thefile)

        run.count = count

        run.epochs = epochs

        run.K = run.currK()

        run.coefficient = run.randcoeff()


    # Initialization

    def currK(run):
        k = []
        index = numpy.random.choice(run.data.shape[0], run.count, replace = False)
        
        for idx in index:

            k.append(run.data[idx])

        return numpy.asarray(k)


    # Calculates a distance list between each data inputs as compared to the KMeans
    
    def euclidlist(run,data):
        list = []
        for idx in range(len(run.K)):

            distance = numpy.linalg.norm(data - run.K[idx])
            list.append(distance)

        return numpy.asarray(list)


    # Observation to each class

    def assignK(run):
        grouping = [[] for _ in range(run.count)]

        for idx in range(run.data.shape[0]):

            distcalc = run.euclidlist(run.data[idx])
            grouping[numpy.argmin(distcalc)].append(run.data[idx])

        return grouping


    # Updates K-Centroid based on current iteration, curr is ther returned value from assignK()

    def updatealgo(run, curr):
        assert (len(run.K) == len(curr)), "Check in update()"

        for idx in range(len(run.K)):

            twodimension = numpy.asarray(curr[idx])
            temp = numpy.mean(twodimension, axis=0)
            
            run.K[idx] = temp

        return run.K


    # Calculates the sum square errors

    def squareerror(run,curr):
        error = 0

        for idx in range(len(run.K)):
            x = numpy.asarray(curr[idx])
            dimension = 0
            for idx2 in range(x.shape[0]):

                dimension += (numpy.linalg.norm(x[idx2]-run.K[idx]))**2

            error += dimension

        return error


    # The plot function for the KMeans

    def theplot(run,curr,epochs):
        epoch = epochs + 1

        for idx in range(len(curr)):
            x = numpy.asarray(curr[idx])

            plot.scatter(x.T[0], x.T[1], cmap=plot.get_cmap('rainbow'))
            plot.scatter(run.K.T[0],run.K.T[1], c='#111111')

        print()
        print()
        print("#" + str(epoch))
        print()
        plot.savefig('Figure_' + str(epoch) + '.png')

    # Random coefficient generation

    def randcoeff(run):
        coefficients = []
        for idx in range(run.data.shape[0]):

            coefficient = numpy.random.uniform(0.0,1.0,(run.count,))
            coefficients.append(coefficient)

        return numpy.asarray(coefficients)


    # The main KMeans functionality

    def kmean(run):
        error,clusters = [],[]
        
        cluster = run.assignK()
        run.theplot(cluster,0)

        for idx in range(run.epochs):

            run.updatealgo(cluster)

            print(run.K)
            print()

            x = deepcopy(run.K)
            clusters.append(x)

            cluster = run.assignK()
            error.append(run.squareerror(cluster))
            run.theplot(cluster,(idx + 1))

        print("The #th iteration & their centroid set with smallest sum(s) of squared error(s): ")
        result = numpy.argmin(numpy.array(error))
        print(result + 2)
        print(clusters[result])


## MAIN PROGRAM OPTIONS / CODE ##

count = 10
epochs = 10

Program3 = kmeans("545_cluster_dataset.txt", count, epochs) # Runs the program

Program3.kmean() # KMeans initiation