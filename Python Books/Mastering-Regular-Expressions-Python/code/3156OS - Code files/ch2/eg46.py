import re

pattern = re.compile("^\w+\: (\w+/\w+/\w+)", re.M)
pattern.findall("date: 12/01/2013⇢\ndate: 11/01/2013")

