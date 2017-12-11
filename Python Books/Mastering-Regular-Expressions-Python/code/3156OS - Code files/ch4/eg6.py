import re

pattern = re.compile(r'John(?!\sSmith)')
result = pattern.finditer("I would rather go out with John McLane than with John Smith or John Bon Jovi")
for i in result:
    print i.start(), i.end()



