import re

pattern = re.compile(r"(\d\d-)?(\w{3,4})(?(1)(-\d\d))")
pattern.match("34-erte-22")
pattern.search("erte")
pattern.match("34-erte")
