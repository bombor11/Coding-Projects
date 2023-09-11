import math

#Softmax activation function , is a function aimed at the output layer

#example of activation function for ilustration

layer_output = [4.8, 1.21, 2.385]

#In theory if we want to predict an output we have to take the highest value of the neuron from the array in this case element 0 = 4.8
#However we want to make sure that we are actually training the model so we will use an activation function to check how wrong is the model


#we have to use an exponential function where y = e^x where e = Euler's number

#This will make sure that we do not have to deal with negative variables while keeping the integrity of the value

#Raw python implementation of exponential function

#E = 2.71828182846

E = math.e

exp_values = []

#the exponential function where it takes all the outputs and use the fomula mentioned above
for output in layer_output:
    exp_values.append(E**output)

print(f"Exponential values: \n {exp_values}")


#The next step it will be to take the values and normalise those
#Normalisation is gonna be a single output neuron values devided by the sum of all the other output neurons in that output layes
#this will give us the probability distribution, we still need to exponentiate before this point because we need to get rid of the negative

#Coding Normalisation after we have done our exponentiate

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(f"Normalised values: \n {norm_values}")
print(f"Sum of all values should results in near 1: \n {sum(norm_values)}")


#Version of this implemention using numpy:

import numpy as np

def num_py_implementation():
    exp_values = np.exp(layer_output)

    norm_values = exp_values / np.sum()

