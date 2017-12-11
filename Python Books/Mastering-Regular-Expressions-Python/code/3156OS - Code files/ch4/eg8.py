import re

pattern = re.compile(r'\d{1,3}(?=(\d{3})+(?!\d))')
results = pattern.finditer('1234567890')
for result in results:
    print result.start(), result.end()





