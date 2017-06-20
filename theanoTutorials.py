#figuring out theano
import numpy
import theano.tensor as T
from theano import function, In, shared, pp, scan
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams 
import theano

#init variables for pretty printing
counter = 0

def incCounter():
    print "\n"
    global counter
    print str(counter) + ":"
    counter = counter + 1

def println(theanoFunc, inputs):
    incCounter()
    print theanoFunc(*inputs)
    print "\n"

def zeroThroughFour():
    println(f0, [2, 3])
    println(f1, [[[1,2], [3, 4]], [[1, 2], [3, 4]]])
    println(f2, [[[1, 2], [3, 4]]])
    println(f3, [[[1, 2], [3, 4]], [[2, 4],[4, 5]]])
    println(f4, [[[1, 2],[3, 4]]])

def five():
    #0 0 1 2 -1
    #variables that are shared can have states and using updates
    incCounter()
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

def six():
    #7 0
    #using the givens parameter to replace a node in a graph for a specific function
    #this allows you to use a function for a certain value of the used shared value (in the function) without changing the value of the shared variable
    incCounter()
    #using 3 for the value of x6
    print f6(1, 3)
    #the value fo x6 has been kept at its original 0
    print x6.get_value()

def seven():
    #100 110
    incCounter()
    #returned is 100 as the shared variable is 100
    print f7(10)
    #the value fo the shared variable post the function call updates to 110
    print x7.get_value()

def eight():
    #two inputs from uniform dist, two of the same prints from normal dist, almost zero
    #demonstrating how to generate random numbers
    incCounter()
    print f8_1()
    print f8_1()
    print "\n"
    print f8_2()
    print f8_2()
    print "\n"
    print f8_3()

def nine():
    #note that one and two are not the same as three and four because of the seed change mid producing the distributions
    incCounter()
    print f9_1()
    print "\n"
    print f9_2()
    print "\n"
    print f9_3()
    print "\n"
    print f9_4()

def ten():
    #the first two are different but the last two are the same
    x10 = MRG_RandomStreams(123)
    y10 = MRG_RandomStreams(235)
    z10 = x10.uniform((2, 2))
    a10 = y10.uniform((2, 2))
    f10_1 = function([], z10)
    f10_2 = function([], a10)
    incCounter()
    print f10_1()
    print "\n"
    print f10_2()
    x10.rstate = y10.rstate
    for (x10_1, y10_1) in zip(x10.state_updates, y10.state_updates):
        x10_1[0].set_value(y10_1[0].get_value())
    b10 = x10.uniform((2, 2))
    c10 = y10.uniform((2, 2))
    f10_3 = function([], b10)
    f10_4 = function([], c10)
    print "\n"
    print f10_3()
    print "\n"
    print f10_4()

def eleven():
    #taking a derivative
    incCounter()
    print f11(4)

def twelve():
    #taking more complex derivative
    incCounter()
    print f12([[1, 0], [-1, -2]])

def thirteenThrough():
    #compute the jacobian
    incCounter()
    print f13([4, 4])
    incCounter()
    print f14([4,4])
    incCounter()
    print f15_1([[1, 2], [1, 3]], [1, 1], [0, 1])
    #note that the jacobian then times v should be the same thing
    print f15_2([[1, 2], [1, 3]], [1, 1], [0, 1])
    #now trying with the lop
    print f15_3([[1, 2], [1, 3]], [0,1])
    print f15_4([[1, 2], [1, 3]], [1, 1], [0, 1])
    #lop and rop can be extended to the hessian by passing in the jacobian or gradient as the first variable of the lop or rop 


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
f6 = function([y6, z6], a6, givens=[(x6, z6)])
#7--copying function f5
x7 = shared(100)
f7 = f5.copy(swap={x5:x7})
#8
x8 = RandomStreams(seed=234)
y8 = x8.uniform((2, 2))
z8 = x8.normal((2, 2))
f8_1 = function([], y8)
f8_2 = function([], z8, no_default_updates=True) #not updating z8.rng
f8_3 = function([], y8 + y8 - 2*y8)
#9--seeding random streams
x9 = RandomStreams(seed=234)
y9 = x9.uniform((2, 2))
z9 = x9.uniform((2, 2))
f9_1 = function([], y9)
f9_2 = function([], z9)
x9.seed(9000)
b9 = x9.uniform((2, 2))
c9 = x9.uniform((2, 2))
f9_3 = function([], b9)
f9_4 = function([], c9)
#10--setting two random number generators equal to each other to produce the same random variables
#ten needed to be included in the function to change the rstate and updates after the first print showing incongruence
#11--working on derivatives in theano
x11 = T.dscalar('x11')
y11 = x11 ** 2
gy11 = T.grad(y11, x11) #derivative of y11 wrt x11
f11 = function([x11], gy11)
#12--specific derivative of the logistic function
x12 = T.dmatrix('x12')
y12 = T.sum(1/(1 + T.exp(-x12)))
gy12 = T.grad(y12, x12)
f12 = function([x12], gy12)
#13--in order to take the jacobian (derivative of matrix wrt matrix) then we need to use scan
#or use theano.gradient.jacobian()
x13 = T.dvector('x13')
y13 = x13 ** 2
#loop through each vector of the matrix and compute the gradient wrt the vector x
z13, a13 = scan(lambda i13, y13, x13 : T.grad(y13[i13], x13), sequences=T.arange(y13.shape[0]), non_sequences=[y13, x13])
f13 = function([x13], z13, updates = a13)
#14--computing the hessian manually
#or use theano.gradient.hessian()
x14 = T.dvector('x14')
y14 = x14 ** 2
z14 = y14.sum()
gy14 = T.grad(z14, x14)
a14, b14 = scan(lambda i14, gy14, x14 :  T.grad(gy14[i14], x14), sequences=T.arange(gy14.shape[0]), non_sequences=[gy14, x14])
f14 = function([x14], a14, updates=b14)
#15--using the R operator to multiply the jacobian by a vector, can also be used for matrix times vector
w15 = T.dmatrix('w15')
x15 = T.dvector('x15')
v15 = T.dvector('v15')
y15 = T.dot(w15, x15)
Jv15_1 = T.Rop(y15, x15, v15)
#this rop means jacobian of y15 wrt x15 then * v15
#note that this operator only needs a dot product between a vector and a matrix, the vector can come out of the jacobian if need be
f15_1 = function([w15, x15, v15], Jv15_1)
a15 = theano.gradient.jacobian(T.dot(w15, x15), x15)
f15_2 = function([w15, x15, v15], T.dot(a15, v15))
#now extend this with the L operator
Jv15_2 = T.Lop(y15, x15, v15)
#interestingly the L operator does not need the input x15
f15_3 = function([w15, v15], Jv15_2)
f15_4 = function([w15, x15, v15], T.dot(v15, a15))







if __name__ =="__main__":
    zeroThroughFour()
    five()
    six()
    seven()
    eight()
    nine()
    ten()
    eleven()
    twelve()
    thirteenThrough()