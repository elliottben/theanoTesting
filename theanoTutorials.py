#figuring out theano
import numpy
import theano.tensor as T
from theano import function

#init variables for pretty printing
counter = 0
def println(theanoFunc, inputs):
    global counter
    print str(counter) + ":"
    print theanoFunc(*inputs)
    print "\n"
    counter = counter + 1

#0
x0 = T.dscalar('x0')
y0 = T.dscalar('y0')
z0 = x0+y0
f0 = function([x0, y0], z0)
#1
x1 = T.dmatrix('x1')
y1 = T.dmatrix('y1')
z1 = x1 + y1
f1 = function([x1, y1], z1)
#2
x2 = T.dmatrix('x2')
y2 = 1/(1+T.exp(-x2))
f2 = function([x2], y2)
#3
x3, y3 = T.dmatrices('x3', 'y3')
z3 = x3 - y3
a3 = abs(z3)
b3 = a3**2
f3 = function([x3, y3], [z3, a3, b3])
#4 Setting a default value for a function

if __name__ =="__main__":
    println(f0, [2, 3])
    println(f1, [[[1,2], [3, 4]], [[1, 2], [3, 4]]])
    println(f2, [[[1, 2], [3, 4]]])
    println(f3, [[[1, 2], [3, 4]], [[2, 4],[4, 5]]])