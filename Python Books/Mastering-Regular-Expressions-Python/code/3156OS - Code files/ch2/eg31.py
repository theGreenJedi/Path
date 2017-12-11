import re

text = "imagine a new *world*, a magic *world*"
pattern = re.compile(r'\*(.*?)\*')
pattern.sub(r"<b>\g<1>1<\\b>", text)

