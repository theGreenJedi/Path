import re

pattern = re.compile(r"""[#|_] + #comment
              \ \# #comment
              \d+""", re.VERBOSE)

pattern.findall("#⇢#2")

