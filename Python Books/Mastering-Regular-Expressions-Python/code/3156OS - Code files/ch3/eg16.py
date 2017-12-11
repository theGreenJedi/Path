import re

pattern = re.compile(r"(\d\d-)?(\w{3,4})-(?(1)(\d\d)|[a-z]{3,4})$")
pattern.match("34-erte-22")
pattern.match("34-erte")
pattern.match("erte-abcd")
