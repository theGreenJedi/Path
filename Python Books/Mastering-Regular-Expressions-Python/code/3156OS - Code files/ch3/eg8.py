import re

pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)")
match = pattern.search("Hello world")
match.group("first")
match.group("second")

