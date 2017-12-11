import re

def normalize_orders(matchobj):
    if matchobj.group(1) == '-': return "A"
    else: return "B"

re.sub('([-|A-Z])', normalize_orders, '-1234 A193 B123')

