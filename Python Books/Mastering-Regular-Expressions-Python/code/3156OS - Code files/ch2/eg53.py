# -*- encoding: utf-8 -*-
import re

re.findall(r"\u03a9", u"adeΩa")
re.findall(ur"\u03a9", u"adeΩa")
u"Ω".encode("utf-8")
re.findall(r'Ω', "adeΩa")
re.findall(r'\xce\xa9', "adeΩa")
re.findall(r'Ω', u"adeΩa")
re.findall(ur'Ω', u"adeΩa")
re.findall(ur"ñ" ,ur"Ñ", re.I)
