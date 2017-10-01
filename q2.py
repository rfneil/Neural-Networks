import numpy as np 
import random
import math 

class Perceptron:

	def __init__(self):
		self.alpha = 10
		np.random.seed(42)
		self.w0 = np.random.rand(3,3)
		
		self.w1 = np.random.rand(1,4)
		
	def sigmoid(self, x, deriv = False):
		sig = 1/(1 + np.exp(-x))
		if deriv == True:	
			return (sig*(1-sig))
		else :
			return sig
	
	def train(self, X, Y, iterations):
		# Training dataset:
		X = np.column_stack(([1,1,1,1],X))
		
		for iter in range(iterations):
			choice = np.random.randint(0, len(X)) #pick a sample index
            		x = X[choice] #the sample from X
            		
			# Forward propagation
			l0 = np.dot(x , np.transpose(self.w0))
			print l0# feed input to layer0
			l1 = self.sigmoid(l0)# layer1 gets output of sigmoid activations
			print "l1", l1
			l1 = l1.reshape(1,3)
			l1 = np.column_stack(([1],l1))
			print "l1a", l1
			inl2 = np.dot(l1 , np.transpose(self.w1))
			l2 = self.sigmoid(inl2) # layer2 gets output of sigmoid activations
			
			# Backward propagation
			l2_error = Y[choice] - l2# compute error for output layer
			l2_delta = l2_error * self.sigmoid(inl2, deriv = True)# compute "delta" for layer2
			l1_error = l2_delta * self.w1# compute error for layer1
			print l1_error
			l1_delta = []
			for j in range(2):
				l1_delta[j] = self.sigmoid(l0[j], deriv = True) * l1_error# compute "delta" values (use error)
			
			
			for i in range(3):
				self.w0[i] =  self.w0[i] - self.alpha*l1_delta[i+1]*x[i]# update weights (use alpha)
			self.w1 =  self.w1 - self.alpha*l2_delta*l1# update weights (use alpha)
			
	def predict(self, X):
		# Testing dataset
		X = np.column_stack(([1,1,1,1],X))
		l0 = np.dot(x,np.transpose(self.w0))# feed input to layer0
		l1 = self.sigmoid(l0)
		l1 = np.column_stack([1],l0)# layer1 gets output of sigmoid activations
		inl2 = np.dot(l1,np.transpose(self.w1))
		l2 = self.sigmoid(inl2)# layer2 gets output of sigmoid activations
		if l2<0.5:
			return 0
		else: 
			return 1 # output of layer2
		

if __name__ =='__main__':
	p = Perceptron()
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	
	y = np.array([[0],[1],[1],[0]]) 
	p.train(X, y, 1000)
	
	print(p.predict(X))

