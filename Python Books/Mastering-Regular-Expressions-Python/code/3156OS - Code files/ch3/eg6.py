import re

pattern = re.compile(r"(\d+)-(\w+)")
pattern.sub(r"\2-\1", "1-a\n20-baer\n34-afcr")

