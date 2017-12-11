import re

pattern = re.compile(r"(\w+) (\w+)")
match = pattern.search("Hello world")
match.group(1)
match.group(2)

