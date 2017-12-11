import re

pattern = re.compile("(\w+) (\w+)")
match = pattern.search("Hello⇢World")
match.groups()

