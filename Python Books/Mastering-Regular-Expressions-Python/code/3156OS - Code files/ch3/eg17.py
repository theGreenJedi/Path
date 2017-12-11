import re

re.findall(r'(a|b)+', 'abaca')
re.findall(r'((?:a|b)+)', 'abbaca')
re.findall(r'(a|b)', 'abaca')
