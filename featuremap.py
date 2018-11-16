import numpy as np
import random

def feature_training(X, Wi, a):
	W1 = Wi[0] + a*(X[0]-Wi[0])
	W2 = Wi[1] + a*(X[1]-Wi[1])
	W3 = Wi[2] + a*(X[2]-Wi[2])
	W4 = Wi[3] + a*(X[3]-Wi[3])
	WNew = [W1, W2, W3, W4]
	return WNew;

def feature_decision(Y):
	if Y[0] <= Y[1] and Y[0] < Y[2] and Y[0] < Y[3]:
		w = 'Waitlisted'
	elif Y[1] < Y[0] and Y[1] <= Y[2] and Y[1] < Y[3]:
		w = 'Accepted'
	elif Y[2] < Y[0] and Y[2] < Y[1] and Y[2] <= Y[3]:
		w = 'Rejected'
	elif Y[3] < Y[0] and Y[3] < Y[1] and Y[3] < Y[2]:
		w = 'Unknown'
	return w;

def feature_computation(X,W):
	Y1 = ((X[0]-W[0,0])**2) + ((X[1]-W[1,0])**2) + ((X[2]-W[2,0])**2) + ((X[3]-W[3,0])**2)
	Y2 = ((X[0]-W[0,1])**2) + ((X[1]-W[1,1])**2) + ((X[2]-W[2,1])**2) + ((X[3]-W[3,1])**2)
	Y3 = ((X[0]-W[0,2])**2) + ((X[1]-W[1,2])**2) + ((X[2]-W[2,2])**2) + ((X[3]-W[3,2])**2)
	Y4 = ((X[0]-W[0,3])**2) + ((X[1]-W[1,3])**2) + ((X[2]-W[2,3])**2) + ((X[3]-W[3,3])**2)
	Y = np.array([Y1, Y2, Y3, Y4])
	return Y;

#main function
if __name__ == "__main__":
	r = random.random()
	W = [[r+4.5, r+4.5, r+4.5, r+4.5],
		[r+4.3, r+4.3, r+4.3, r+4.3],
		[r+2.3, r+2.3, r+2.3, r+2.3],
		[r+5.5, r+5.5, r+5.5, r+5.5]]

	S1 = [6.2,4.5,3.3,6.1]
	S2 = [4.9,4.5,3.2,6.4]
	S3 = [5.5,7.3,3.7,8.4]
	S4 = [2.3,3.4,2.4,3]
	S5 = [6.5,6.3,3.7,8.9]
	S6 = [5.4,6,3.4,7.1]
	S7 = [7.5,6.7,3.9,9.6]
	S8 = [7.1,6.5,3.7,9.1]
	S9 = [5.1,3.6,3.2,6.7]
	S10 = [4.9,4.8,3.1,5.9]
	S11 = [4.5,7.3,3.4,7.7]
	S12 = [3.8,4.1,2.7,5]
	S13 = [6,4.5,3.2,6.4]
	S14 = [4.3,3.9,2.9,4.8]
	S15 = [5.7,5.7,3.7,8.5]
	S16 = [6.1,6.9,3.6,8.9]

	array_students = np.array([S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16])

	alpha = 0.6

	for epoch in range(0,29):
		for x in range(0,15):
			n = feature_decision(feature_computation(array_students[x],W))
			W = feature_training(array_students[x], W, alpha)
		alpha = alpha*0.75

	print(W)

	print(feature_decision(feature_computation(S1, W)))
	print(feature_decision(feature_computation(S2, W)))
	print(feature_decision(feature_computation(S3, W)))
	print(feature_decision(feature_computation(S4, W)))
	print(feature_decision(feature_computation(S5, W)))
	print(feature_decision(feature_computation(S6, W)))
	print(feature_decision(feature_computation(S7, W)))
	print(feature_decision(feature_computation(S8, W)))
	print(feature_decision(feature_computation(S9, W)))
	print(feature_decision(feature_computation(S10, W)))
	print(feature_decision(feature_computation(S11, W)))
	print(feature_decision(feature_computation(S12, W)))
	print(feature_decision(feature_computation(S13, W)))
	print(feature_decision(feature_computation(S14, W)))
	print(feature_decision(feature_computation(S15, W)))
	print(feature_decision(feature_computation(S16, W)))
