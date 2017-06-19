#completing the logistic regression example of the tutorial
import sys
import theano
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import function, printing, shared

#the number of datapoints per features
N = 400
#number of features
feats = 784
#my random numbers
rng = np.random
#generate dataset
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high = 2))
training_steps = 10000

#declaring theano x and y
x = T.dmatrix("x")
y = T.dvector("y")

#init the weight vector
w = theano.shared(rng.randn(feats), name="w")
#init the bias vector
b = theano.shared(0., name="b")

"""
print ("Initial model:")
print (w.get_value())
print (b.get_value())
"""

#construct the graph
p_1 = 1/ (1+ T.exp(-T.dot(x, w) - b)) #using sigmoid activation function
prediction = p_1 > 0.5 #if greater than 0.5 guess true if not guess false
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) #using the cross entropy loss function (logistic)
cost = xent.mean() + 0.01*(w**2).sum() #interesting cost function--make sure that the mean of the error isn't too high as well as hold the weights at bay with regularization technique
gw, gb = T.grad(cost, [w, b]) #computing the gradient of the cost function wrt weights and biases--very cool and powerful step

#compile the graph
train = function(inputs=[x, y], outputs=[prediction, xent], updates=((w, w-0.1*gw), (b, b-0.1*gb))) #update w and b on each regression while traversing the neuron, activation function , and loss function
predict = function(inputs=[x], outputs=prediction)

#train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    if i %100 ==0:
        print str(i/100) + " " + str(err.mean())

"""
print("Final model:")
print w.get_value()
print b.get_value()
"""
print "target values for D:"
print (D[1])
print "prediction on D:"
print predict(D[0])

#Note that by removing the regularization will allow the err to decrease, yet for now increasing due to  + 0.01*(w**2).sum() term.