import re

data = "aaaaabbbbbaaaaccccccdddddaaa"
regex.match("(\w+)-\d", data)
regex.match("(?>\w+)-\d", data)

