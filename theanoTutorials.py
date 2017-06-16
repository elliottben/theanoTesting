#figuring out theano
import numpy
import theano.tensor as T
from theano import function
from theano import In
from theano import shared

#init variables for pretty printing
counter = 0
def println(theanoFunc, inputs):
    global counter
    print str(counter) + ":"
    print theanoFunc(*inputs)
    print "\n"
    counter = counter + 1

def zeroThroughFour():
    println(f0, [2, 3])
    println(f1, [[[1,2], [3, 4]], [[1, 2], [3, 4]]])
    println(f2, [[[1, 2], [3, 4]]])
    println(f3, [[[1, 2], [3, 4]], [[2, 4],[4, 5]]])
    println(f4, [[[1, 2],[3, 4]]])

def five():
    #0 0 1 2 -1
    #variables that are shared can have states and using updates
    print str(counter) + ":"
    print x5.get_value()
    #now increment the value, note that returned is the value of x5 as specified in function
    print f5(1)
    #if run again then value of x5 changes to 1
    print f5(1)
    #checking the value we see it is at 2 because the function has been run twice with a counting value fo 1
    print x5.get_value()
    #this value can be reset with the set_value() function
    x5.set_value(-1)
    print x5.get_value()
    print "\n"

def six():
    #7 0
    #using the givens parameter to replace a node in a graph for a specific function
    #this allows you to use a function for a certain value of the used shared value (in the function) without changing the value of the shared variable
    print str(counter) + ":"
    #using 3 for the value of x6
    print b6(1, 3)
    #the value fo x6 has been kept at its original 0
    print x6.get_value()
    print "\n"


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
#4
x4, y4 = T.dmatrices('x4', 'y4')
z4 = x4 + y4
f4 = function([x4, In(y4, value=[[1, 1], [1, 1]])], z4)
#5
x5 = shared(0)
y5 = T.iscalar('y5')
f5 = function([y5], x5, updates=[(x5, x5+y5)])
#6
x6 = shared(0)
y6 = T.iscalar('y6')
z6 = T.scalar(dtype=x6.dtype)
a6 = x6 * 2 + y6
b6 = function([y6, z6], a6, givens=[(x6, z6)])

if __name__ =="__main__":
    zeroThroughFour()
    five()
    six()