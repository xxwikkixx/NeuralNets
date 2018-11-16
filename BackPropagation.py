import math
import numpy as np

#HW 11 BackPropagation
def NeuronOutputBackProp(Vb, V1, V2, X1, X2):
    return sigmoidFunc((Vb + (V1*X1) + (V2*X2)));

#signoid activation function formula
def sigmoidFunc(z):
    return 1/(1 + math.exp(-z));

def errorCalc(t, y):
    return 0.5 * ((target[t] - y) ** 2);

def backPropCalculation(t, y):
    return (target[t] - y)*y*(1-y);

def backPropHidden(z, delta, w):
    return z*(1-z)*delta*w;

#main function
if __name__ == "__main__":
    v = np.matrix('1 -1; ' #00, 01
                  '2 2; '  #10, 11
                  '-2 -1') #20, 21

    w = np.matrix('-1; ' #0
                  '2; '  #1
                  '-2')  #2

    alpha = 0.5
    epoch = 1

    x1 = 0
    x2 = 0
    t = 0

    xColl = np.array([[0,0],
                      [0,1],
                      [1.0],
                      [1,1],
                      ])
    target = np.array([[0],
                       [1],
                       [1],
                       [0],
                       ])


    for j in range(epoch):
        print("\n---Epoch ", j+1)

        for i in range(4):
            ###### 4 iterations to train the networkk ######
            z3 = NeuronOutputBackProp(v[0,0], v[1,0], v[2,0], x1, x2)
            z4 = NeuronOutputBackProp(v[0,1], v[1,1], v[2,1], x1, x2)
            y5 = sigmoidFunc(w[0] + (w[1]*z3) + (w[2]*z4))

            print("Z3 = ", z3)
            print("Z4 = ", z4)
            print("Y5 = ", y5)