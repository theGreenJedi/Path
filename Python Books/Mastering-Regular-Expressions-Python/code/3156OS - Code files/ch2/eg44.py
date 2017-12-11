import re

pattern = re.compile(r"[a-z]+", re.I)
pattern.search("Felix")
pattern.search("felix")

