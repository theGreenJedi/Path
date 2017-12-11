Python 3.2.3 (v3.2.3:3d0686d90f55, Apr 10 2012, 11:25:50) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> dir(list)
['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
>>> colours = []
>>> prompt = 'Enter another one of your favourite colours (type return to end): '
>>> 
>>> colour = input(prompt)
Enter another one of your favourite colours (type return to end): blue
>>> colour
'blue'
>>> colours
[]
>>> while colour != '':
	colours.append(colour)
	colour = input(prompt)

	
Enter another one of your favourite colours (type return to end): yellow
Enter another one of your favourite colours (type return to end): brown
Enter another one of your favourite colours (type return to end): 
>>> colours
['blue', 'yellow', 'brown']
>>> colours.extend(['hot pink', 'neon green'])
>>> colours
['blue', 'yellow', 'brown', 'hot pink', 'neon green']
>>> colours.pop()
'neon green'
>>> colours
['blue', 'yellow', 'brown', 'hot pink']
>>> colours.pop(2)
'brown'
>>> colours
['blue', 'yellow', 'hot pink']
>>> colours.remove('black')
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    colours.remove('black')
ValueError: list.remove(x): x not in list
>>> if colours.count('yellow') > 0:
	colours.remove('yellow')

	
>>> colours
['blue', 'hot pink']
>>> if 'yellow' in colours:
	colours.remove('yellow')

	
>>> colours
['blue', 'hot pink']
>>> colours.extend('auburn', 'taupe', 'magenta')
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    colours.extend('auburn', 'taupe', 'magenta')
TypeError: extend() takes exactly one argument (3 given)
>>> colours.extend(['auburn', 'taupe', 'magenta'])
>>> colours
['blue', 'hot pink', 'auburn', 'taupe', 'magenta']
>>> colours.sort()
>>> colours
['auburn', 'blue', 'hot pink', 'magenta', 'taupe']
>>> colours.reverse()
>>> colours
['taupe', 'magenta', 'hot pink', 'blue', 'auburn']
>>> colours.insert(-2, 'brown')
>>> colours
['taupe', 'magenta', 'hot pink', 'brown', 'blue', 'auburn']
>>> colours.index('neon green')
Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    colours.index('neon green')
ValueError: 'neon green' is not in list
>>> if 'hot pink' in colours:
	where = colours.index('hot pink')
	colours.pop(where)

	
'hot pink'
>>> colours
['taupe', 'magenta', 'brown', 'blue', 'auburn']
>>> 
