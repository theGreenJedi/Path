import re

pattern = re.compile(r'\bfoo\b')
pattern.match("foo bar")

