''' This is my first implementation of MLP built from Scratch, 
to understand the basics of how a neural network works on the ground.
This implementation is Simple and experimental to comprehend the basis of an nn system '''

''' It includes the manual implementation of forward & backward propagation, gradient descend,
then it involves the training of the model with dummy dataset generated with numpy to make some prediction at the last '''

''' My shortcomings here is using a sigmoid function in hidden layers while the initial requirement is linear.
Though it works fine for this particular example of x+y not for every problem''' # keep in mind our model doesnt know what addition is, it is just predicting the result

'''This whole setup is on random numbers so... they do not have much meaning behind them. except in few scenarios'''

import numpy as np
from random import random




class MLP(object):
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2): #constructor
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers =[self.num_inputs] + self.num_hidden +[self.num_outputs]

        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])* 0.1
            weights.append(w)
        self.weights = weights


        activations = []
        for i in range(len(layers)):
            a = np.zeros((layers[i]))
            activations.append(a)

        self.activations = activations


        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        
        self.derivatives = derivatives
        


    def forward_propagate(self, inputs):
        #it calculates activates
        activations = inputs

        self.activations[0] = inputs
        for i, w in enumerate(self.weights):
            #calculation of netinput xW(1)
            net_inputs= np.dot(activations, w)

            if i == len(self.weights) - 1:
                activations = net_inputs       # linear output
            else:
                activations = self._sigmoid(net_inputs)

            self.activations[i + 1] = activations
            
            #well i+1 cuz a_3 = sigmoid(h_3) and h_3 is well.. = a_2 * w_2 so here i = 2 then activation clearly of a_3 thus i+1


        return activations

    def _sigmoid_derivative(self, x):
        return x*(1-x)

    def backward_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            #now the error grdient was dE/dW2 = (a3 - y) sig'(h3)a2
            #and dE/dW1 = (a3 - y) * sig'(h3) * w2 * sig'(h2) * x 
            #well on gen casde dE/dW_i= (y-a_[i+1]) * sig'(h_[i+1]) * a_i
            # sig'(h_[i+1])= sig(h_[i+1])(1-sig(h_[i+1]))
            # sig(h_[i+1])= a_[i+1]
        
            activations = self.activations[i+1]
            #delta = error*sigmoid derivative first part of dE/dW
            delta = error * self._sigmoid_derivative(activations)

            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i]= np.dot(current_activations_reshaped, delta_reshaped)
            #and dE/dW_[i-1] = (a_[i+1] - y) * sig'(h_[1+i]) * w[i] * sig'(h[i]) * a[i-1] 
            error = np.dot(delta, self.weights[i].T) #(a_[i+1] - y) * sig'(h_[1+i]

            if verbose:
                print("derivatives for w{}: {}".format(i, self.derivatives[i]))

        return error


    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            
            derivatives = self.derivatives[i]
            weights += derivatives* learning_rate


    #train method
    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                
                    #calculate the error
                error = target - output
                    #back propagation
                self.backward_propagate(error)
                
                
                    #apply gradient descent
                self.gradient_descent(learning_rate)
                sum_error += self._mse(target, output)
                  

    def _mse(self, target, output):
        return np.average((target-output)**2)


    def _sigmoid(self, x):
        return (1.0 / (1 + np.exp(-x)))

if __name__=="__main__":
    #creating MLP
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    mlp = MLP(2, [5], 1)


    #train our mlp
    mlp.train(inputs, targets, 50, 0.1)
    
    
    #create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output=mlp.forward_propagate(input)
    print()

    print()
    print("Our network beleives that {} + {} is equal to {}".format(input[0], input[1], output[0]))


        

        
