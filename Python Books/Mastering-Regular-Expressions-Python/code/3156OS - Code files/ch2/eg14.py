import re

pattern = re.compile(r'^<HTML>', re.MULTILINE)
pattern.search("<HTML>")
pattern.search(" <HTML>")
pattern.search("  \n<HTML>")

