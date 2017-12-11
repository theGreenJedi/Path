import re

from time import clock as now

def test(f, *args, **kargs):
    start = now()
    f(*args, **kargs)
    print "The function %s lasted: %f" %(f.__name__, now() - start)

pattern = re.compile(r'/Hello\s(World|Continent|Country)')
def optimized():
    pattern.match("Hello\sCountry")

def callonethousandtimes():
    for _ in range(1000):
        optimized()

test(callonethousandtimes)
