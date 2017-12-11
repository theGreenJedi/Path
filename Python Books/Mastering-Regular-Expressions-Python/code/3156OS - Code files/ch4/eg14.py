import re

pattern = re.compile(r'\w+\s[\d-]+\s[\d:,]+\s(.*(?<!authentication\s)failed)')
pattern.findall("INFO 2013-09-17 12:13:44,487 authentication failed")
pattern.findall("INFO 2013-09-17 12:13:44,487 something else failed")








