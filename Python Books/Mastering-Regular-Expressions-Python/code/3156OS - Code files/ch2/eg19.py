import re

pattern = re.compile(r"(\w+) (\w+)")
pattern.findall("Hello world hola mundo")

