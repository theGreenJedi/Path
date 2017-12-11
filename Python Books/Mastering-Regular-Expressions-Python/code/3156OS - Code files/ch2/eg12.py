import re

pattern = re.compile(r'<HTML>$')
pattern.match("<HTML> ", 0,6)
pattern.match("<HTML> "[:6])

