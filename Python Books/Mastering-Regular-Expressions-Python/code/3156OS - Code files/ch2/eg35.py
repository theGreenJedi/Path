import re

pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)")
match = pattern.search("Hello world")
match.group('first')
match.group(1)
match.group(0, 'first', 2)

