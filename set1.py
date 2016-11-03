import numpy as np
import random


class PLA:
    ''' Class for the Perceptron Learning Algorithm
    '''

    def __init__(self, n):
        ''' Generates a random target function f, and generates
            n random data points x from [-1, 1] x [-1, 1]. The outputs f(x)
            are stored into the list 'signs'. The weight vector w is
            initialized to be all zeros.
        '''
        x1, x2, y1, y2 = [np.random.uniform(-1.0, 1.0) for i in range(4)]
        self.target = np.array([y2-y1, x1-x2, x2*y1 - x1*y2])
        self.data = np.array([(np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0), 1.0) for i in range(n)])
        self.signs = np.array([int(np.sign(self.target.transpose().dot(x)))
                      for x in self.data])
        self.w = np.array([0.0, 0.0, 0.0])

    def runPLA(self):
        ''' Runs the pla to generate w that estimates f
        '''

        w = np.array([0.0, 0.0, 0.0])
        h = np.array([int(np.sign(w.transpose().dot(x)))
            for x in self.data])  # signs generated from w

        misclass = []
        # get the misclassified points
        for i, element in enumerate(h):
            if (element != self.signs[i]):
                misclass.append([i, self.data[i]])

        iterations = 0
        while misclass:
            length = len(misclass)
            randIndex = random.randint(0, length - 1)
            index = misclass[randIndex][0]
            w += self.signs[index] * misclass[randIndex][1]  # update w
            h = [int(np.sign(w.transpose().dot(x)))
                      for x in self.data]
            misclass = []  # update misclassified points again
            for i, element in enumerate(h):
                if (element != self.signs[i]):
                    misclass.append([i, self.data[i]])
            iterations += 1

        self.w = np.array(w)
        return iterations

    def getError(self):
        ''' Get error by generating 1000 new random points and seeing how many
            outputs w(x) do match f(x).
        '''
        newData = np.array([(np.random.uniform(-1.0, 1.0),
                    np.random.uniform(-1.0, 1.0), 1.0) for i in range(1000)])
        original = np.array([int(np.sign(self.target.transpose().dot(x)))
                          for x in newData])
        wSign = np.array([int(np.sign(self.w.transpose().dot(x)))
                          for x in newData])
        correct = 0.0
        for i, element in enumerate(original):
            if element == wSign[i]:
                correct += 1.0
        return 1 - (correct/1000.0)

def getAverageIter(n, points):
    ''' Get the average number of iterations for n runs of the pla
        with N = points
    '''
    total = 0
    for i in range(n):
        pla = PLA(points)
        total += pla.runPLA()
    average = total // n
    print(average)
    return average

def getAverageError(n, points):
    ''' Get the average error for n runs of the pla with N = points
    '''
    total = 0.0
    for i in range(n):
        pla = PLA(points)
        pla.runPLA()
        total += pla.getError()
    average = total / n
    print(average)
    return average

getAverageIter(1000, 10)
getAverageError(10, 10)
getAverageIter(1000, 100)
getAverageError(10, 100)




