import re

pattern = re.compile(r"[0-9]+")
pattern.sub("-", "order0 order1 order13")

