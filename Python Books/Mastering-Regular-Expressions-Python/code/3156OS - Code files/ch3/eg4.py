import re

pattern = re.compile(r"(\d+)-\w+")
it = pattern.finditer(r"1-a\n20-baer\n34-afcr")
match = it.next()
match.group(1)
match = it.next()
match.group(1)
match = it.next()
match.group(1)

