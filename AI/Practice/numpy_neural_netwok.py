import numpy as np

#in this file I will be using the numpy library to simplify and reduce the bolierplate of the code
#numpy offers different funtions for calculation.

def neuron_with_np():
    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]

    bias = 2

    #the funtion dot multiplies each element with other and adds it toghater e.g. w0*i0+.... + bias
    output = np.dot(weights, inputs) + bias
    return output

def layer_of_neurons_with_np():
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    
    bias = [2, 3, 0.5]

    #Since now I am working with a layer of neurons it is important to pas the parammeters to the dot funtion in the right oder
    #we pass the weights first since the first parrameter determins how the array will be indexed 
    #the funtion will do the same thing however it will output 3 resaults since it will add the the inputs to each neuron and there are 3 neurons in total 
    output = np.dot(weights, inputs) + bias
    return output

neuron_output = neuron_with_np()
print(f"Single neuron calculation using numPy: {neuron_output}")

layer_output = layer_of_neurons_with_np()
print(f"Layer of neurons calculation using numPy: {layer_output}")