import re
import locale
locale.setlocale(locale.LC_ALL, '')
chars = ''.join(chr(i) for i in xrange(256))
" ".join(re.findall(r"\w", chars, re.LOCALE))
