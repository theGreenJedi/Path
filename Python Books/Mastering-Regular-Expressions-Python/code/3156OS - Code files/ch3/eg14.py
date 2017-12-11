# -*- encoding: utf-8 -*-
import re

re.findall(r"(?u)\w+" ,ur"ñ")
re.findall(r"\w+" ,ur"ñ", re.U)
