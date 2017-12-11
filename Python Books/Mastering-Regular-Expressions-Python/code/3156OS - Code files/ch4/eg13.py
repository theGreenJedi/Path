import re

pattern = re.compile(r'(?<!John\s)Doe')
results = pattern.finditer("John Doe, Calvin Doe, Hobbes Doe")
for result in results:
    print result.start(), result.end()







