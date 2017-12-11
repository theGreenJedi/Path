import re

pattern = re.compile(r"(\w+) \1")
match = pattern.search(r"hello hello world")
match.groups()

