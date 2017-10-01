import numpy as np
import random

class Perceptron:
    def __init__(self):
        self.weights = np.random.rand(3) #randomize the weights
    
    def train(self, X, Y, iterations):
    	learning_rate=0.01
        for i in range(iterations):
            choice = np.random.randint(0, len(X)) #pick a sample index
            x = X[choice] #the sample from X
            v = np.dot(x,self.weights) #compute v
            if v<0:
            	result=0
            else:
            	result=1
            expected = Y[choice] #its expected value from y
            error = expected - result #compute error
            self.weights = self.weights + learning_rate*error*x #update weights
        print "Weights are : ",self.weights

    def predict(self,x):
    	result = np.dot(x,self.weights)#compute v
    	for i in range(len(x)):
        	if(result[i]<0):
        		output = 0
        	else:
        		output = 1
        	print output

if __name__=='__main__':
        #AND input (extra leading 1's in the X vectors might come in handy when multiplying with weights)
    	X = np.array([[0,0],[0,1],[1,0],[1,1]])
        X = np.column_stack(([1,1,1,1],X))
        Y = np.array([0,0,0,1])
        iterations = 1000

        perc = Perceptron()

        perc.train(X,Y,iterations)
        
	print "Output for AND gate"
	perc.predict(X)
        
        
        
        
