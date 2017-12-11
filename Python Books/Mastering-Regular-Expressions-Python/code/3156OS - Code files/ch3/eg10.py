import re

pattern = re.compile(r"(?P<word>\w+) (?P=word)")
match = pattern.search(r"hello hello world")
match.groups()
