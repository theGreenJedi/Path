import re
from time import clock as now

def test(f, *args, **kargs):
    start = now()
    f(*args, **kargs)
    print "The function %s lasted: %f" %(f.__name__, now() - start)

def alternation(text):
    pat = re.compile('spa(in|niard)')
    pat.search(text)

test(alternation, "spain")
