import math
import numpy as np

#HW 11 BackPropagation
def NeuronOutputBackProp(Vb, V1, V2, X1, X2):
    return sigmoid((Vb + (V1*X1) + (V2*X2)));

#signoid activation function formula
def sigmoid(z):
    return 1/(1 + math.exp(-z));

def errorCalc(t, y):
    return 0.5 * ((target[t] - y) ** 2);

def backPropCalculation(t, y):
    return (target[t] - y)*y*(1-y);

def backPropHiddenCalc(z, delta, w):
    return z*(1-z)*delta*w;

def backPropCalc(t,y):
    return (target[t] - y)*y*(1-y)

#main function
if __name__ == "__main__":
    epoch_num = 1
    alpha = 0.5

    # Our Internal neuron matrix   (Given)
    w03, w04 = 1, -1
    w13, w14 = 2, 2
    w23, w24 = -2, -1

    # Our Output neuron matix (Given)
    w05 = -1
    w35 = 2
    w45 = -2


    x1 = 0
    x2 = 0
    t = 0

    xColl = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1],
                      ])

    target = np.array([[0],
                       [1],
                       [1],
                       [0],
                       ])

    for j in range(epoch_num):

        # Number of iterations
        for i in range(4):

            z3 = sigmoid(w03 + (xColl[i][0] * w13) + (xColl[i][1] * w23))
            z4 = sigmoid(w04 + (xColl[i][0] * w14) + (xColl[i][1] * w24))

            # Calculate sigmoid A.F for Output layer
            y5 = sigmoid(w05 + (z3 * w35) + (z4 * w45))

            # Print progress so far
            print("\nSigmoid values : ")
            print("Z3 = ", z3)
            print("Z4 = ", z4)
            print("Y5 = ", y5)

            print("\nErrors : ")
            # Total Error
            print("Total Error: ", errorCalc(i, y5)[0])

            d5 = backPropCalc(i, y5)[0]


            d3 = backPropHiddenCalc(z3, d5, w35)
            d4 = backPropHiddenCalc(z4, d5, w45)

            # Printing
            print("Back Propagation Error d5: ", d5)
            print("Back Propagation Error d4: ", d4)
            print("Back Propagation Error d3: ", d3)

            # Bias Changes
            chg_w03 = alpha * d3
            chg_w04 = alpha * d4
            chg_w05 = alpha * d5

            # Weight Changes (Output layer)
            chg_w35 = alpha * d5 * z3
            chg_w45 = alpha * d5 * z4

            # Weight Changes (Hidden Layer)
            chg_w13 = alpha * d3 * xColl[i][0]
            chg_w14 = alpha * d4 * xColl[i][0]
            chg_w23 = alpha * d3 * xColl[i][1]
            chg_w24 = alpha * d4 * xColl[i][1]

            print("\nChanges in weights: ")
            print("change in w03 = ", chg_w03)
            print("change in w04 = ", chg_w04)
            print("change in w05 = ", chg_w05)
            print("change in w13 = ", chg_w13)
            print("change in w14 = ", chg_w14)
            print("change in w23 = ", chg_w23)
            print("change in w24 = ", chg_w24)
            print("change in w35 = ", chg_w35)
            print("change in w45 = ", chg_w45)

            # Bias Adjustments
            w03 = w03 + chg_w03
            w04 = w04 + chg_w04
            w05 = w05 + chg_w05

            # Weight Adjustments
            w13 = w13 + chg_w13
            w14 = w14 + chg_w14
            w23 = w23 + chg_w23
            w24 = w24 + chg_w24
            w35 = w35 + chg_w35
            w45 = w45 + chg_w45


            print("\nOutput Layer Matrix after iteration", i + 1)
            print("|", '{:>6s}'.format("%.3f" % w05), "|")
            print("|", '{:>6s}'.format("%.3f" % w35), "|")
            print("|", '{:>6s}'.format("%.3f" % w45), "|")

