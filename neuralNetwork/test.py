import numpy
import matplotlib
import neuralNetwork.neuralNetworkTest as nn


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Train the neural network
# Go through all records in the training data set

for record in training_data_list:
    # Split the record by the ',' commas
    all_values = record.split(',')

    # Scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01

    # Create the target output values
    targets = numpy.zeros(output_nodes) + 0.01

    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass