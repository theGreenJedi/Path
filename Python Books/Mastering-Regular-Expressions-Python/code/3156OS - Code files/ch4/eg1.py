import re

pattern = re.compile(r'fox')
result = pattern.search("The quick brown fox jumps over the lazy dog")
print result.start(), result.end()
