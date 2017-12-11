import re

def alternation(text):
    pat = re.compile('spa(in|niard)')
    pat.search(text)

import cProfile
cProfile.run("alternation('spaniard')")
