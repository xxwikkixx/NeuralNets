import numpy as np
#Hamming Network

def ham_computation(X,W,b):
	Y1 = (X[0]*W[0,0])+(X[1]*W[1,0])+(X[2]*W[2,0])+b
	Y2 = (X[1]*W[0,1])+(X[1]*W[1,1])+(X[2]*W[2,1])+b
	Y3 = (X[2]*W[0,2])+(X[1]*W[1,2])+(X[2]*W[2,2])+b
	Y4 = (X[3]*W[0,3])+(X[1]*W[1,3])+(X[2]*W[2,3])+b
	Y = np.array([Y1, Y2, Y3, Y4]);
	#print(Y)
	return Y;

def ham_Decison_Output(Y):
	if Y[0] >= Y[1] and Y[0] > Y[2]:
		w = 'Accept'
	elif Y[1] >= Y[0] and Y[1] >= Y[2] or Y[0] == Y[2]:
		w = 'Waitlist'
	elif Y[2] > Y[0] and Y[2] > Y[1]:
		w = 'Reject'
	return w;

#main function
if __name__ == "__main__":
	accepted = [1,1,1,1]
	waitlisted = [1,1,1,-1]
	rejected = [1,-1,-1,-1]

	weights = np.array([accepted, waitlisted, rejected])*.5
	bias = 2

	S1 = [1,-1,-1,-1]
	S2 = [-1,-1,-1,-1]
	S3 = [1,1,1,1]
	S4 = [-1,-1,-1,-1]
	S5 = [1,1,1,1]
	S6 = [-1,1,-1,1]
	S7 = [1,1,1,1]
	S8 = [1,1,1,1]
	S9 = [-1,-1,-1,1]
	S10 = [-1,-1,-1,-1]
	S11 = [-1,1,-1,1]
	S12 = [-1,-1,-1,-1]
	S13 = [1,-1,-1,-1]
	S14 = [-1,-1,-1,-1]
	S15 = [1,1,1,1]
	S16 = [1,1,1,1]

	print(weights)

	print(ham_Decison_Output(ham_computation(S1, weights, bias)))
	print(ham_Decison_Output(ham_computation(S2, weights, bias)))
	print(ham_Decison_Output(ham_computation(S3, weights, bias)))
	print(ham_Decison_Output(ham_computation(S4, weights, bias)))
	print(ham_Decison_Output(ham_computation(S5, weights, bias)))
	print(ham_Decison_Output(ham_computation(S6, weights, bias)))
	print(ham_Decison_Output(ham_computation(S7, weights, bias)))
	print(ham_Decison_Output(ham_computation(S8, weights, bias)))
	print(ham_Decison_Output(ham_computation(S9, weights, bias)))
	print(ham_Decison_Output(ham_computation(S10, weights, bias)))
	print(ham_Decison_Output(ham_computation(S11, weights, bias)))
	print(ham_Decison_Output(ham_computation(S12, weights, bias)))
	print(ham_Decison_Output(ham_computation(S13, weights, bias)))
	print(ham_Decison_Output(ham_computation(S14, weights, bias)))
	print(ham_Decison_Output(ham_computation(S15, weights, bias)))
	print(ham_Decison_Output(ham_computation(S16, weights, bias)))
