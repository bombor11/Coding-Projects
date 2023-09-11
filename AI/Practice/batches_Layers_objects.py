import numpy as np

np.random.seed(0)

#This file will focus on creating the actual neural network where we will use baches insted of single points of data , and objects to create the network

def bach_of_data_one_layer():

    #now that we want to work in a more practicall way then teoretical we want to make a input into a batch of data where each array of inputs is a reading or data at a point
    #each array elemment is a featuer of that data.
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    
    #since I am only modifing the input data the weights and biases will stay the same cuz the neurons are not modified
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    
    bias = [2, 3, 0.5]

    #For the out put we want to do a matrix multiplication where we multiplay the first colum with the first row and get a new matrix with the resaults

    #in this situation since weights and inputs have a shape problem we have to use Transpose too make the rown of the weight matrix into rows so the .dot function will be able to do the calculation

    #to do this all we have to do is to conver the weights into a numpy array and use the .T function

    output = np.dot(inputs, np.array(weights).T) + bias

    return output

def two_layers():
    #this method will show the calculation of a neural network with 2 layers of neurons, this is hard coded just for visualisation howevr this will be transformed into object for simplicity and scalability


    inputs = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

    #first layer of neurons    
    weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
    
    bias = [2, 3, 0.5]
    #second layer of neurons
    weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33,],
            [-0.44, 0.73, -0.13]]
    
    bias2 = [-1, 2, -0.5]

    #here I calculate the first layer which then becomes the inputs for the layers 2
    layer1_output = np.dot(inputs, np.array(weights).T) + bias 

    layer2_outputs = np.dot(layer1_output, np.array(weights2).T) +bias2

    return layer2_outputs  

bach_of_outputs = bach_of_data_one_layer()
print(f"A bach of data calculated using matrix multiplication(in numpy) only one layer: \n {bach_of_outputs} \n")

two_layers_outputs = two_layers()
print(f"The calculation of two layers of neurons: \n {two_layers_outputs}")


print(f"\n ------------------------------ \n")

#for the initialisation we want to make sure that the data stays between a small range usually between -1 and plus 1 to avoid extension of memory
#The weights will be initialised to random values
#The biases needs to be initialise in case some of the neurons outputs 0 and it will propagte to the network making the network value*0 which results in 0 kak dead network


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons) # we need to define the shape size of the inputs comming in this layer 
        self.bias = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


def using_object_to_create_nn():

    #Inputs or training data is ussually named capital X
    X = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

    layer1 = Layer_Dense(4,5) #first param we pass is the size of our featuers in the inputs in this case 4 
    layer2 = Layer_Dense(5,2) #sec layer needs to have the shape of the first layer since that number of neurons will feed data in the new layer

    layer1.forward(X) #I pass the data in the first layer
    layer2.forward(layer1.output) #I pass the first's layer resaults in the secound layer 

    print(f"First Layer: \n {layer1.output} \n")
    print(f"Second Layer: \n {layer2.output} \n")

using_object_to_create_nn()