#pyton document to be the lstm
import numpy as np
import theano
from theano import tensor as T
from theano import function, scan, shared
from theano.tensor.shared_randomstreams import RandomStreams
import collections


class Neuron:
    """
    class to define the neuron being used
    """
    def linear(self, x, batch_size, dimensions, n, w, b, precision):
        """
        x = the input into the linear function \n
        batch_size = array size of the initial batch of input \n
        dimensions = list of tuple dimensions for the neuron(s) layer \n
        n = the tuple in the dimensions that specifies the weight matrix \n
        w = the w unique shared variable for grad descent \n
        b = the b unique shared variable for grad descent \n
        precision = string, how precise the calculations should be \n
        """
        #configuring the random weights to be (-1/sqrt(noi), 1/sqrt(noi))
        #reference: https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        #reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html#numpy.random.uniform
        lower = -np.sqrt(1./(reduce(lambda x, y:x*y, dimensions[n][:-1])))
        higher = np.sqrt(1./(reduce(lambda x, y:x*y, dimensions[n][:-1])))
        #multiply all dimensions
        w_np = np.random.uniform(lower, higher, dimensions[n]).astype(precision)
        w.set_value(w_np)
        #first portion of the multiplication
        linear_1 = T.dot(x, w)
        #configuring random biases
        #configure the same random way as the weights
        map(lambda x:batch_size.append(x), dimensions[n][1:])
        b_np = np.random.uniform(lower, higher, batch_size).astype(precision)
        b.set_value(b_np)
        linear = linear_1 + b
        return linear



class Activation:
    """
    class to define all the activation functions that can be used
    """
    def sigmoid(self, neuron_type):
        """
        neuron_type = the type of neuron, currently is just linear \n
        """
        sigmoid = T.dmatrix("sigmoid")
        sigmoid = 1/(1+ T.exp(-(neuron_type)))
        return sigmoid


class Loss:
    """
    least_squared: This is the least squared cost function \n
    logistic_loss: Implementation of logistic loss funciton \n
    """
    def logistic_loss(self, y, y_hat):
        logistic_loss = T.dmatrix("logistic_loss")
        logistic_loss = -y * T.log(y_hat) - (1-y) * T.log(1-y_hat)
        return logistic_loss


class Cost:
    """
    cost function is composed of the loss function as well as regularization that can be added
    """
    def __init__(self, loss, w_list, b_list):
        """
        init the cost function with a loss function
        """
        self.loss = loss
        self.cost = loss
        self.w_list = w_list
        self.b_list = b_list

    def current(self):
        return self.cost

    def update_parameters(self, learning_rate):
        """
        update the shared parameters \n
        w = a list of the weights starting from the foremost layers \n
        b = a list of biases starting from the foremost layers \n
        learning_rate = the rate at which the program learns (can be function) \n
        """
        updates = collections.OrderedDict()
        for w_i, b_i in zip(self.w_list, self.b_list):
            gw_i, gb_i = T.grad(T.mean(self.loss), [w_i, b_i])
            updates[w_i] = (w_i - learning_rate*gw_i)
            updates[b_i] = (b_i - learning_rate*gb_i)
        return updates



class Neural_network:
    """
    layers = number of hidden layers in the model \n
    layers_size = the number of neurons at each layer \n
    activation = the specific activation unit used for the model \n
    learning_rate = the rate at which the model takes steps down the cost function \n
    cost_function = the cost function of the network \n
    """
    def __init__(self, learning_rate, batch_size, dimensions, precision):
        #configure theano variables
        self.precision = precision
        #configure basic parameter variables
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dimensions = dimensions
        #necessary elements
        self.neuron = Neuron()
        self.activation = Activation()
        self.activation = Activation()
        self.w_list = []
        self.b_list = []
        #overall building function
        self.func = T.dmatrix("pre-compiled_function")

    def fully_connected_network(self, x):
        for layer in xrange(len(self.dimensions)):
            w_name = "w" + str(layer)
            b_name = "b" + str(layer)
            w = theano.shared(np.array([[]]).astype(self.precision), name=w_name)
            b = theano.shared(np.array([[]]).astype(self.precision), name=b_name)
            self.w_list.append(w)
            self.b_list.append(b)
            #add the neuron to the function
            if (layer==0):
                self.func = self.neuron.linear(x, [self.batch_size], self.dimensions, layer, w, b, self.precision)
            else:
                self.func = self.neuron.linear(self.func, [self.batch_size], self.dimensions, layer, w, b, self.precision)
            #add the activation to the function
            self.func = self.activation.sigmoid(self.func)
        return self.func

    def logistic_loss_update(self, y):
        loss = Loss()
        cost = Cost(loss.logistic_loss(y, self.func), self.w_list, self.b_list)
        return cost.update_parameters(self.learning_rate)

    def print_function_graph(self, file):
        theano.printing.pydotprint(self.func, outfile=file, var_with_name_simple=True)

class Help:
    """
    class you can instantiate if you need help!
    """
    def example_neural_network(self):
        """
        DATA
        [
            [0.1, 0.3, 0.5]
            [0.3, 0.2, 0.1]
            [0.4, 0.5, 0.4]
        ]
        """
        #first declare the input matrix parameters in theano
        x = T.dmatrix("x")
        y = T.dmatrix("y")
        #specify the dimensions of each of the weight matrices
        dimensions = [[2, 10], [10, 5], [5, 1]]
        #declare the nn with (learning_rate, dimensions, accuracy)
        network = Neural_network(0.1, 3, dimensions, 'float64')
        #print the network framework out to a file
        network.print_function_graph("pydotprint.png")
        my_func = function([x, y], network.fully_connected_network(x), updates = network.logistic_loss_update(y))
        #train the function
        for i in xrange(1000):
            my_func([[0.1, 0.3], [0.3, 0.2], [0.4, 0.5]], [[0.5], [0.1], [0.4]])
        #print results
        print my_func([[0.1, 0.3], [0.3, 0.2], [0.4, 0.5]], [[0.5], [0.1], [0.4]])
        
class mean_pooling:
    """
    Description to be filed
    """

class Albert:
    """
    Albert should be composed of an lstm, mean pooling, and then a neural network
    """


if __name__ == "__main__":
    help = Help()
    help.example_neural_network()