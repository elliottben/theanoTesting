#testing different code segments of the bot
import theano
from theano import tensor as T, function, shared, scan
import numpy as np
import collections
import sklearn
import sklearn.decomposition as decomp

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

a_real = [[1.0, 2.0, 3.0]]
b_real = [[4.0, 5.0, 6.0]]
c_real = [[7.0, 8.0, 9.0]]
a_np = np.array(a_real)
b_np = np.array(b_real)
c_np = np.array(c_real)
a_theano = T.dmatrix("a_theano")
b_theano = T.dmatrix("b_theano")
c_theano = T.dmatrix("c_theano")
result = function([a_theano, b_theano, c_theano], T.concatenate([a_theano, b_theano, c_theano], axis=0))
my_sequence = [a_real, b_real, c_real]
print result(*my_sequence)
result_1 = function([], T.concatenate(my_sequence, axis=1))
print "result 1"
print result_1()

input_1 = [9]
input_2 = [8]
print input_1 + input_2

updates = collections.OrderedDict()
update_1 = collections.OrderedDict()
update_2 = collections.OrderedDict()
update_1["key1"] = "key1"
update_2['key2'] = "key2"
updates.update(update_1)
updates.update(update_2)
print update_1
print update_2
print updates
print "\n"

x_1 = T.dmatrix("x_1")
def f(x):
    x_v = T.dvector("x_v")
    x_v = x[0]
    return x_v

func = function([x_1], f(x_1))
print func([[ 0.5], [-0.1], [ 0.4]])

def tensor_split_into(x, pieces):
        axis=0
        split_dim = x.shape[axis]/pieces
        split_distribution = []
        for piece in xrange(pieces):
            split_distribution.append(split_dim)
        return theano.tensor.split(x, split_distribution, pieces, axis=axis)

func = function([x], tensor_split_into(x, 4))
array = np.array([[1, 2, 3, 4]])
print array[0]


value = 3
print "EXPERIMENT"
print value
a = [[3, 4, 5, 8], [1, 2, 5, 3]]
print np.array(a).shape
my_pca = decomp.PCA(n_components=value)
my_pca.fit(a)

a_new = my_pca.transform(a)
print np.array(a_new).shape
print a_new