import re

pattern = re.compile(r"(\w+) (\w+)")
it = pattern.finditer("Hello world hola mundo")
match = it.next()
match.groups()
match.span()

match = it.next()
match.groups()
match.span()

