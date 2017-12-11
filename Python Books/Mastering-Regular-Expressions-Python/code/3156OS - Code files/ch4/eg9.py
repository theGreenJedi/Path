import re

pattern = re.compile(r'\d{1,3}(?=(\d{3})+(?!\d))')
pattern.sub(r'\g<0>,', "1234567890")






