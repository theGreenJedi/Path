import re

text = "imagine a new *world*, a magic *world*"
match = re.search(r'\*(.*?)\*', text)
match.expand(r"<b>\g<1><\\b>")

