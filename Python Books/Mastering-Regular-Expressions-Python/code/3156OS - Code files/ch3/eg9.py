import re

pattern = re.compile(r"(?P<country>\d+)-(?P<id>\w+)")
pattern.sub(r"\g<id>-\g<country>", "1-a\n20-baer\n34-afcr")

