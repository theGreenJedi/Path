import re

from time import clock as now

def test(f, *args, **kargs):
    start = now()
    f(*args, **kargs)
    print "The function %s lasted: %f" %(f.__name__, now() - start)

pattern = re.compile(r'\bfoo\b')
def reuse():
    pattern.match("foo bar")

def callonethousandtimes():
    for _ in range(1000):
        reuse()

test(callonethousandtimes)





