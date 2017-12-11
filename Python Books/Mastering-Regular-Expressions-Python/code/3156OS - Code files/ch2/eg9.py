import re

pattern = re.compile(r'^<HTML>')
pattern.match("<HTML>")

pattern.match("  <HTML>",  2)

