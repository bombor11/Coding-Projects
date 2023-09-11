#In this file it shows how the calculation of a neuron and a layer of neuron is done without any libraries.


#method that calculates a single neuron that recives 4 inputs
def neuron():
    #define the inputs aka the data and the wights
    #inputs are the data that comes in the neuron this data might be the actual data or might be inputs from neurons from previous layer
    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]

    #evrey unique neuron has a unique bias that it is used as an offset
    bias = 2

    #the callculation for the neuron is done by using the folloing formula InputN1 * WeightN1 + InputNn * WeightNn + Bias
    output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
    return output


#method that calculates the output layer where there are 4 inputs but 3 different neurons
def outputLayer():
    inputs = [1, 2, 3, 2.5]

    #since in this layers there are 3 neurons there will be an array of arrays where each sub-array
    #represents a neuron with it's inputs weights. This is a Matrix since it is 2 by 2 
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    
    #each neuron has it's own bias since are 3 neurons we will have an array with 3 values
    bias = [2, 3, 0.5]

    layer_output = []

    #those are nested loops that calculates the each neuron inputs and weights.
    #the zip function combines the 2 lists where the frist element of the list is another list
    #then it appends each resut in a new list after calculation
    for neuron_weights, neuron_bias in zip(weights, bias):
        neuron_output = 0
        for n_inputs, weights in zip(inputs, neuron_weights):
            neuron_output += n_inputs*weights
        neuron_output += neuron_bias
        layer_output.append(neuron_output)

    return layer_output


single_neuron = neuron()
print(f"Single neuron calculation: {single_neuron}")


layer_of_neurons = outputLayer()
print(f"Layer of neurons calculation: {layer_of_neurons}")

#important notes:
#Shapes shows the dimension
#[1,2,3] has the shape of (3,) since it has 3 elements and it is a 1d array or Vector
#[[1,2,3]
# [4,2,1]] has the shape of (2,3) since it has 2 rows aka 2 lists and each list has 3 elements and it is a 2d array or Matrix
