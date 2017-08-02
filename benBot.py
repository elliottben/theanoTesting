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

    def lstm(self, x, h_t_1, c_t_1, w, u, b, vector_dimensions, precision):
        """
        a unit returning a time step output. The following is an explanation

        input information is added to cell information based on the previous time step of that cell to produce the new cell state.  
        This is designed around the idea that the cell "chooses" to make a decision based off 
        previous information and "state" or current information and "state"

        the ouput is then computed "typically" with the activation output of a normal linear unit being multiplied by the
        tanh of the produced state for the cell
        """
        activation = Activation()
        #forget, input, output, and C in between values
        fioC_t = []
        c_t = T.dmatrix("c_t")
        h_t = T.dmatrix("h_t")
        lower = -np.sqrt(1./(vector_dimensions))
        higher = np.sqrt(1./(vector_dimensions))
        for w_i, u_i, b_i in zip(w, u, b):
            w_np = np.random.uniform(lower, higher, np.array([vector_dimensions, vector_dimensions])).astype(precision)
            w_i.set_value(w_np)
            u_np = np.random.uniform(lower, higher, np.array([vector_dimensions, vector_dimensions])).astype(precision)
            u_i.set_value(u_np)
            b_np = np.random.uniform(lower, higher, np.array([1, vector_dimensions])).astype(precision)
            b_i.set_value(b_np)
            fioC_t.append(activation.options["logistic_function"](T.dot(x, w_i) + T.dot(h_t_1, u_i) + b_i))
        c_t = fioC_t[0]*c_t_1 + fioC_t[1]*fioC_t[2]
        h_t = fioC_t[3]*c_t
        return h_t, c_t

class Activation:
    """
    class to define all the activation functions that can be used
    """
    def __init__(self):
        self.options = {
            "logistic_function" : self.logistic_function,
            "inverse_tangent_function" : self.inverse_tangent_function,
            "linear_function" : self.linear_function,
            "relu_function" : self.relu_function
        }
    
    def logistic_function(self, neuron_type):
        """
        neuron_type = the type of neuron, currently is just linear \n
        """
        logistic_function = T.dmatrix("logistic_function")
        logistic_function = 1./(1.+ T.exp(-(neuron_type)))
        return logistic_function

    def inverse_tangent_function(self, neuron_type):
        """
        neuron_type = the type of neuron, currently is just linear \n
        """
        inverse_tangent_function = T.dmatrix("inverse_tangent_function")
        inverse_tangent_function = 1.7159 * T.tanh(1.5 * (neuron_type))
        return inverse_tangent_function

    def linear_function(self, neuron_type):
        return neuron_type

    def relu_function(self, neuron_type):
        return T.nnet.relu(neuron_type)

class Loss:
    """
    least_squared_loss: This is the least squared cost function \n
    logistic_loss: Implementation of logistic loss funciton \n
    """

    def __init__(self):
        self.options = {
            "logistic_loss":self.logistic_loss,
            "least_squared_loss":self.least_squared_loss
        }
        self.loss_function = T.dmatrix("loss_function")

    def logistic_loss(self, y, y_hat):
        self.loss_function = -y * T.log(y_hat) - (1.0-y) * T.log(1.0-y_hat)
        return self.loss_function

    def least_squared_loss(self, y, y_hat):
        self.loss_function = T.sum(0.5*((y-y_hat)**2))
        return self.loss_function

class Cost:
    """
    cost function is composed of the loss function as well as regularization that can be added
    """
    def __init__(self, loss, w_list, b_list, u_list, h_t_1, network_components):
        """
        init the cost function with a loss function
        """
        self.loss = loss
        self.cost = loss
        self.w_list = w_list
        self.b_list = b_list
        #the following is for lstm
        self.u_list = u_list
        self.h_t_1 = h_t_1
        #the structure of the network
        self.network_components = network_components

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
        if("lstm_chain" in self.network_components):
            for w_i, b_i, u_i, h_t_1_i in zip(self.w_list, self.b_list, self.u_list, self.h_t_1):
                gw_i, gu_i, gb_i = T.grad(T.mean(self.loss), [w_i, u_i, b_i])
                updates[w_i] = (w_i - learning_rate*gw_i)
                updates[b_i] = (b_i - learning_rate*gb_i)
                updates[u_i] = (u_i - learning_rate*gu_i)
        elif ("fully_connected_network" in self.network_components):
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
    def __init__(self, learning_rate, precision):
        #configure theano variables
        self.precision = precision
        #configure basic parameter variables
        self.learning_rate = learning_rate
        self.loss = Loss()
        #necessary elements
        self.neuron = Neuron()
        self.activation = Activation()
        self.w_list = []
        self.u_list = []
        self.b_list = []
        self.h_t_1_list = []
        self.func_inputs_x = []
        self.func_inputs_y = []
        self.func_inputs_y_compiled = None
        #function componenets
        self.network_components = []
        #overall building function
        self.func = T.dmatrix("pre-compiled_function")
        self.func_compiled = None

    def printWeights(self):
        print "\nWeights:"
        print "length: " + str(len(self.w_list))
        for item in self.w_list:
            print "\nlayer" + str(item) + "\n"
            print item.get_value() 

    def printBiases(self):
        print "\nBiases:"
        print "length: " + str(len(self.b_list))
        for item in self.b_list:
            print "\nlayer" + str(item) + "\n"
            print item.get_value()

    def fully_connected_network(self, batch_size, dimensions, x_input=None, activation_type="inverse_tangent_function"):

        #configure the class variables for the neural network

        #add a fully connected output to the layers
        self.network_components.append("fully_connected_network")
        #create the symbolic x and y
        x = T.dmatrix("x_input")
        if (x_input is not None):
            x = x_input
        else:
            self.func_inputs_x.append(x)
        y = T.dmatrix("y")
        #define the x and y inputs for theano
        self.func_inputs_y = [y]
        #define the compiled y input for the loss function
        self.func_inputs_y_compiled = y
        #set the other parameters for the algorithm
        self.batch_size = batch_size
        self.dimensions = dimensions

        #begin to code the layers of the network

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
            self.func = self.activation.options[activation_type](self.func)
        return self.func

    def lstm_chain(self, vector_dimensions, chain_length, precision, x_input=None):
        #add lstm_chain to the layers
        self.network_components.append("lstm_chain")
        #create outputs that are the same dimensions as x
        self.vector_dimensions = vector_dimensions
        prev_h = np.zeros([1, self.vector_dimensions])
        prev_C = np.zeros([1, self.vector_dimensions])
        for link in xrange(chain_length):
            #each neuron gets one input and one output, unless that input is already provided, then use that input
            x = T.dvector("x" + str(link))
            if (x_input is not None):
                x = x_input[link]
            #store symbolic variables
            self.func_inputs_x.append(x)
            #each lst neuron has four w, u, b
            link_w = []
            link_u = []
            link_b = []
            for i in xrange(4):
                w_name = "w" + str(link) + str(i)
                u_name = "u" + str(link) + str(i)
                b_name = "b" + str(link) + str(i)
                link_w.append(theano.shared(np.array([[]]).astype(self.precision), name=w_name))
                link_u.append(theano.shared(np.array([[]]).astype(self.precision), name=u_name))
                link_b.append(theano.shared(np.array([[]]).astype(self.precision), name=b_name))
                self.w_list.append(link_w[i])
                self.u_list.append(link_u[i])
                self.b_list.append(link_b[i])
            prev_h, prev_C = self.neuron.lstm(x, prev_h, prev_C, link_w, link_u, link_b, self.vector_dimensions, precision)
            self.h_t_1_list.append(prev_h)
        #create the output theano variables, clear the current list
        self.func_inputs_y = []
        for link in xrange(chain_length):
            y = T.dvector("y" + str(link))
            self.func_inputs_y.append(y)
        #create the compiled function inputs
        self.func_inputs_y_compiled = T.concatenate(self.func_inputs_y, axis=0)
        #return every time step output concatenated
        self.func = T.concatenate(self.h_t_1_list, axis=1)
        return self.func

    def update_loss(self, type="least_squared_loss"):
        cost = Cost(self.loss.options[type](self.func_inputs_y_compiled, self.func), self.w_list, self.b_list, self.u_list, self.h_t_1_list, self.network_components)
        return cost.update_parameters(self.learning_rate)

    def return_compiled_func(self, type="least_squared_loss"):
        updates = self.update_loss(type)
        self.func_compiled = function((self.func_inputs_x + self.func_inputs_y), self.func, updates=updates)
        return self.func_compiled

    def print_loss(self, inputs, type="least_squared_loss"):
        loss = function((self.func_inputs_x + self.func_inputs_y), T.mean(self.loss.options[type](self.func_inputs_y_compiled, self.func)))
        print loss(*inputs)

    def print_function_graph(self, file, function):
        theano.printing.pydotprint(function, outfile=file, var_with_name_simple=True)

    def print_network_information(self):
        for layer in self.network_components:
            print layer
        print "\n"

    def tensor_split_into(self, x, pieces):
        axis=1
        split_dim = x.shape[axis]/pieces
        split_distribution = []
        for piece in xrange(pieces):
            split_distribution.append(split_dim)
        return theano.tensor.split(x, split_distribution, pieces, axis=axis)
        

class Help:
    """
    class you can instantiate if you need help!\n
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
        #first declare the data to go in the function
        x_in = [[0.1, 0.3], [0.3, 0.2], [0.4, 0.5]]
        y_out = [[0.5], [-0.1], [0.4]]
        #specify the dimensions of each of the weight matrices
        dimensions = [[2, 5], [5, 1]]
        #declare the nn with (learning_rate, dimensions, accuracy)
        network = Neural_network(0.01, 'float64')
        #create the fully connected layer
        network.fully_connected_network(3, dimensions)
        #print the network framework out to a file
        my_func = network.return_compiled_func()
        #train the function
        network.printWeights()
        network.printBiases()
        print "\nresult:\n"
        print my_func(x_in, y_out)
        for i in xrange(500):
            my_func(x_in, y_out)
            network.print_loss([x_in, y_out])
        network.printWeights()
        network.printBiases()
        print "\nresult:\n"
        print my_func(x_in, y_out)

    def example_lstm_network(self):
        #first define the variables and the data
        x_in = [[0.1, 0.3], [0.6, 0.6]]
        y_out = [[0.2, 0.5], [0.1, 0.1]]
        #print the network function
        network = Neural_network(1.0, 'float64')
        my_lstm_chain = network.lstm_chain(2, len(x_in), 'float64')
        #print the function graph
        network.print_function_graph("pydotprint.png", my_lstm_chain)
        #train and print the output
        lstm_func = network.return_compiled_func()
        network.print_network_information()
        print lstm_func(*(x_in + y_out))
        for i in xrange(100000):
            lstm_func(*(x_in + y_out))
            if (i%5000==0):
                network.print_loss(x_in + y_out)
        print lstm_func(*(x_in + y_out))
        network.print_loss(x_in+y_out)

    def example_nn_to_lstm(self):
        #first define the variables and the data
        x_in = [[0.1, 0.3], [0.6, 0.6]]
        y_out = [[0.2, 0.5, 0.1, 0.1]]
        dimensions = [[4, 8], [8, 4]]
        #print the network function
        network = Neural_network(0.1, 'float64')
        #add lstm
        my_lstm_chain = network.lstm_chain(2, len(x_in), 'float64')
        #add neural network
        network.fully_connected_network(1, dimensions, my_lstm_chain)
        #return function
        my_func = network.return_compiled_func()
        print my_func(*(x_in + [y_out]))
        for i in xrange(500):
            my_func(*(x_in + [y_out]))
            network.print_loss(x_in + [y_out])
        print my_func(*(x_in + [y_out]))

        
        
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
    #help.example_neural_network()
    #help.example_lstm_network()
    help.example_nn_to_lstm()
    #create all the inputs
    """
    x = T.dmatrix("x")
    y = T.dmatrix("y")
    x_in = [[0.1, 0.3], [0.6, 0.6]]
    y_out = [[0.2, 0.5], [0.1, 0.1]]
    #create the function
    network = Neural_network(0.5, 'float64')
    my_lstm_chain = network.lstm_chain(2, len(x_in), 'float64')
    #insert output into neural network dense layer
    dimensions = [[2, 5], [5, 1]]
    my_func = function([x, y], network.fully_connected_network(my_lstm_chain, 3, dimensions), updates = network.update_loss(y, "least_squared_loss", "neural_network"))
    #train the function
    for i in xrange(500):
        my_func(x_in, y_out)
        network.print_loss(x, y, x_in, y_out)
    print my_func(x_in, y_out)
    """
