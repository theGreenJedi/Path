import re

re.search("Espana|ol", "Espanol")
re.search("Espana|ol", "Espana")
re.search("Espana|ol", "ol")
re.search("Espan[aol]", "Espanol")
re.search("Espan[aol]", "Espana")
re.search("Espan[a|ol]", "Espano")

re.search("Espan(a|ol)", "Espana")
re.search("Espan(a|ol)", "Espanol")
re.search("Espan(a|ol)", "Espan")
re.search("Espan(a|ol)", "Espano")
re.search("Espan(a|ol)", "ol")

