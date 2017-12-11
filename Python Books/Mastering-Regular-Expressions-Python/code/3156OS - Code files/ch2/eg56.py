# -*- encoding: utf-8 -*-
import re

re.findall(r"\w+", "هذا⇢مثال")
re.findall(r"\w+", "هذا⇢مثال word", re.ASCII)
