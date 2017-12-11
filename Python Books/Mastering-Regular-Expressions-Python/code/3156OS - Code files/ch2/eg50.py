# -*- encoding: utf-8 -*-
import re

re.findall("\w+", "this is an example")
re.findall(ur"\w+", u"这是一个例子", re.UNICODE)
re.findall(ur"\w+", u"هذا⇢مثال", re.UNICODE)

