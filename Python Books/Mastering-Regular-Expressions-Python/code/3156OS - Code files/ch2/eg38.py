import re

pattern = re.compile(r"(?P<first>\w+) (?P<second>\w+)")
pattern.search("Hello world").groupdict()

