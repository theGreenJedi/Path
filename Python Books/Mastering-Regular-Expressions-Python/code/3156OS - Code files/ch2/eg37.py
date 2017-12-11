import re

pattern = re.compile("(\w+) (\w+)?")
match = pattern.search("Hello ")
match.groups("mundo")
match.groups()
