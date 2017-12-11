import re

pattern = re.compile(r"world")
pattern.search("hello world")
pattern.search("hola mundo ")

