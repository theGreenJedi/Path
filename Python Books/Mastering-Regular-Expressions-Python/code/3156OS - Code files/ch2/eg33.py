import re

text = "imagine a new *world*, a magic *world*"
pattern = re.compile(r'\*(.*?)\*')
pattern.subn(r"<b>\g<1><\\b>", text)

