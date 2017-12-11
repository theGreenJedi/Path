import re

pattern = re.compile(r'<HTML>')
pattern.match("<HTML>"[:2])
pattern.match("<HTML>", 0, 2)

