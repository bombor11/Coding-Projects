import numpy as np
import Create_data
#There are multiple types of activation funtions however for this I will be using a step function
#The activation function comes in plkay after the neuron calculation.
#For this instance depending on the inputs and biases the step function will output a one if the input is > 0 or a 0 if the input is < 0

#A Sigmoid activation function is much more benefical to train a neural network.
#in a sigmoid funtion we get a more specific output in form of a float number.

#Using the step funtion wil limit the ability to calculate the accuracy of thr newral network and to improve it.

#Rectifng liniar activation function when the output is < 0 it will output a 0 however when the output is > 0 it will provide a detailed number not just 1.

np.random.seed(0)

X = [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outupt =[]

#here we check if the inputs are grater then 0 and if they are then it will be appended in the list if they are lower then 0 will be appended
for i in inputs:
    if i > 0:
        outupt.append(i)
    elif i<= 0:
        outupt.append(0)

print(outupt)


X, y = Create_data.create_data(100, 3)

#Object to create the nueral network
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons) # we need to define the shape size of the inputs comming in this layer 
        self.bias = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

#Object for activation function.

class Activation_ReLU:
    def forward(self, inputs):
        self.output =np.maximum(0, inputs)
    
layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)