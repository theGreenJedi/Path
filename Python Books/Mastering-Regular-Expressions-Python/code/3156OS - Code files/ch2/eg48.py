import re

chars = ''.join(chr(i) for i in xrange(256))
" ".join(re.findall(r"\w", chars))

