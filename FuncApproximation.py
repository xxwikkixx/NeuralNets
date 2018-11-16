import numpy as np
import matplotlib.pyplot as plt
import math

#Computes the Bipolar sigmoid function
def bipolarSigmoid(z):
    sigmoid = (2 / (1 + math.exp(-z))) - 1
    return sigmoid;

#calculates the value of z for the middle layer
def Zin(x, i):
    return (x * V[i][0]) + V[i][1];

#main function
if __name__ == "__main__":

    #Values of X used to plot the grah
    intputX = np.array([[-1], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5], [-0.4],
                        [-0.3], [-0.2], [-0.1], [0], [0.1], [0.2], [0.3],
                        [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1]])

    #Weights of the cells
    xWeights = np.array([1.12,
                          2.46,
                          6.11,
                          -1.08,
                          0.96,
                          -1.03,
                          -0.58,
                          -1.11,
                          1.13,
                          1.05])
    #bias of the cells
    bias = np.array([0.36,
                  0.27,
                  0.09,
                  0.28,
                  0.24,
                  -0.29,
                  0.12,
                  -0.34,
                  0.05,
                  0.06])

    #both arrays combined
    V = np.array([[1.12, 0.36],
                 [2.46, 0.27],
                 [6.11, 0.09],
                 [-1.08, 0.28],
                 [0.96, 0.24],
                 [-1.03, -0.29],
                 [-0.58, 0.12],
                 [-1.11, -0.34],
                 [1.13, 0.05],
                 [1.05, 0.06],
                  ])

    w = np.array([[-1.35],
                  [0.14],
                  [4.26],
                  [1.18],
                  [-1.02],
                  [1.20],
                  [0.55],
                  [1.37],
                  [-1.27],
                  [-1.20]])
    wBias = 0.45

    #number of neurons in the middle layer
    neurons = 10
    #lists for the numbers to plot
    listX = []
    listY = []

    for x in range (intputX.size):
        sum = 0
        #calculate the sum of the middle layer sigmoid neuron outputs
        for i in range(neurons):
            sum = sum + (bipolarSigmoid(Zin(intputX[x][0], i)) * w[i][0])

        print("Output layer: ")
        sum = sum + wBias
        listX.append(intputX[x][0])
        listY.append(bipolarSigmoid(sum))

        plt.plot(listX, listY)
        plt.xlabel("Input X")
        plt.ylabel("Network Output")
        plt.show()
