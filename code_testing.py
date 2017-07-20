#testing different code segments of the bot
import theano
from theano import tensor as T, function, shared, scan
import numpy as np

x = T.dmatrix('x')
func = function([x], x.sum()/(x.flatten().shape[0]))

print func([[1, 2], [3, 4]])

my_tuple = (1, 2, 3, 4)
print reduce(lambda x, y:x*y, my_tuple)
print my_tuple[1]

y1 = T.dmatrix('y1')
func_1 = function([y1, x], y1.reshape([4, 1]))
print func_1([[1, 2], [3, 5]], [[1, 2, 3, 4]])

w = theano.shared([1, 2, 3, 4], name="w")
w = theano.shared([[1, 2], [3, 4, 5]], name = "w")
print w.set_value([[3]])
print w.get_value()

a = [1, 2 , 3]
b = [3, 4, 5]
c_1 = a[:-1]
c_2 = b[1:]
map(lambda x:c_1.append(x), c_2)
print c_1

bb = theano.shared((np.array([2,3,4])).astype('float64'))
r = T.dmatrix("r")
function_1 = function([r], T.dot(bb, r))
print function_1([[5],[6],[8]])

shared_matrix_1 = theano.shared((np.array([2, 3, 4])).astype('float64'))
shared_matrix_2 = theano.shared((np.array([2, 3, 4])).astype('float64'))
shared_total = theano.shared(np.array([shared_matrix_1, shared_matrix_2]).astype('float64'))
print shared_total.get_value()
shared_arr = T.dvector("shared_arr")
my_arr = [shared_matrix_1, shared_matrix_2]
results, updates = theano.scan(fn=lambda prior_result, my_arr:prior_result+shared_arr,
                                outputs_info=(np.array([2, 3, 4])).astype('float64'),
                                non_sequences=None,
                                sequences=shared_total)
output_function = function([], results, updates=updates)
print output_function()