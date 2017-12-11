import re

from time import clock as now

def test(f, *args, **kargs):
    start = now()
    f(*args, **kargs)
    print "The function %s lasted: %f" %(f.__name__, now() - start)

def catastrophic(n):
    print "Testing with %d characters" %n
    pat = re.compile('(a+)+c')
    text = "%s" %('a' * n)
    pat.search(text)

for n in range(20, 30):
    test(catastrophic, n)




