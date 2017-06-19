#python testing print
from theano import tensor as T, function, printing
x = T.dvector()
y = T.dvector()
f = function([x], printing.Print('hello world')(x))
f(y)