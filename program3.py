# CS 445, Summer 2021 - Programming Assignment 3 - Dan Jang
# Assignment #3: K-Means Algorithm

## UPDATED WITH DIFFERENTIAL VALUES OF K-CLUSTERS & ITERATIVE ALGORITHMIC CYCLES & Sum of Square Errors Displayed with Each Algorithmic Iteration Result

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

                dimension += (numpy.linalg.norm(x[idx2] - run.K[idx]))**2

            error += dimension

        return error


    # The plot function for the KMeans

    def theplot(run,curr,epochz):
        epoch = epochz
        epoch2 = (epoch + 1)
        for idx in range(len(curr)):
            x = numpy.asarray(curr[idx])

            plot.scatter(x.T[0], x.T[1], cmap=plot.get_cmap('rainbow'))
            plot.scatter(run.K.T[0],run.K.T[1], c='#111111')
            
#        print()
#        print("For K = " + str(run.count) + "...")
#       print("...Where our current iteration # is: " + str(epoch2))
#        print()
        
        plot.savefig('K_' + str(run.count) + '_Figure_Iteraction_' + str(epoch2) + '.png') # UPDATED: To differentiate the labeling based on clusters (K)

    # Random coefficient generation

    def randcoeff(run):
        coefficients = []
        for idx in range(run.data.shape[0]):

            coefficient = numpy.random.uniform(0.0,1.0,(run.count,))
            coefficients.append(coefficient)

        return numpy.asarray(coefficients)


    # The main KMeans functionality

    def kmean(run):
        error = []
        clusters = []
        
        cluster = run.assignK()
        run.theplot(cluster,0)

        for idx in range(run.epochs):

            print()
            print("For K = " + str(run.count) + "...")
            print("...Where our current iteration # is:", idx+1)
            print()
            run.updatealgo(cluster)

            print(run.K)
            print()

            x = deepcopy(run.K)
            clusters.append(x)

            cluster = run.assignK()
            kerror = run.squareerror(cluster)

            # UPDATE: Displays Sum of Error for each chosen model of k
            print("---")
            print("[K = ", run.count, "]", "Current Sum of Square Error for iteration #",  idx+1,  ":", kerror)
            print("---")

            error.append(kerror)
            run.theplot(cluster,(idx))

      #  print("The #th iteration & their centroid set with the smallest sum(s) of squared error(s) within this specific cycle of runs: ")
        result = numpy.argmin(numpy.array(error))
        result2 = (result + 1)
      #  print("...Where our current iteration # is:", result2)
        print()
        print("* * *")
        print()
        print("RESULTS - For K = " + str(run.count) + "; we find that iteration: " + str(result2) + " had the smallest sum of square errors")
      #  print("...Where our current iteration # is:", idx+1, "")
        print(str(result2) + "th iteration's centroid set:")
      #  run.updatealgo(cluster)
        print(clusters[result])
        print()
        print("...with a Sum of Square Error of:")
        print(error[result])
        print()
        print("* * *")


## MAIN PROGRAM OPTIONS / CODE ## UPDATED WITH DIFFERENTIAL VALUES OF K-CLUSTERS & ITERATIVE ALGORITHMIC CYCLES

# count = 10 # Clusters (K) # UPDATED: K is directly inputed as the second parameter of kmeans(x, y, z)
# epochs = 9 # Iterations # UPDATED: iterations are directly inputed as third parameter of kmeans(x, y, z)

Program3ten = kmeans("545_cluster_dataset.txt", 10, 10) # Runs the program with k,# of runs = 10 
Program3ten.kmean() # KMeans initiation with k = 10, # of runs = 10

Program3nine = kmeans("545_cluster_dataset.txt", 9, 9) # Runs the program with k,# of runs = 9
Program3nine.kmean() # KMeans initiation with k = 9, # of runs = 9

Program3five = kmeans("545_cluster_dataset.txt", 5, 5) # Runs the program with k,# of runs  5
Program3five.kmean() # KMeans initiation with k = 5, # of runs = 5

Program3two = kmeans("545_cluster_dataset.txt", 2, 2) # Runs the program with k,# of runs  2
Program3two.kmean() # KMeans initiation with k = 2, # of runs = 2
