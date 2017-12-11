import re

text = "imagine a new *world*, a magic *world*"
pattern = re.compile(r'\*(.*?)\*')
pattern.sub(r"<b>\g<1>1<\\b>", text)

pattern.sub(r"<b>\g11<\\b>", text)


