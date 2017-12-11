import re

pattern = re.compile(r"(\w+) (\w+)")
match = pattern.search("Hello world")

match.group()
match.group(0)
match.group(1)
match.group(2)
match.group(3)
match.group(0, 2)
