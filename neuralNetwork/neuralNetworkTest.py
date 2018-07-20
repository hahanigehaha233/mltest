import numpy
import scipy.special


class NeuralNetwork:

    # Initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set number of nodes in each input,hidden, output,layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Learning rate
        self.lr = learningrate

        # Link weight matrices
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Activation function is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    # Train the neural network
    def train(self, inputs_list, targets_ist):
        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_ist, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs* (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # Query the neural network
    def query(self, inputs_list):
        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

