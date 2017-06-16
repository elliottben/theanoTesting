#figuring out theano
import numpy
import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
f = function([x, y], z)

if __name =="__main__":
    print f(2, 3)