__author__ = 'Chetan'

class Singleton:
    
    __instance = None
    
    def __init__(self):
        if not Singleton.__instance:
            print(" __init__ method called..")
        else:
            print("Instance already created:", self.getInstance())
    
    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = Singleton()
        return cls.__instance


s = Singleton() ## class initialized, but object not created
print("Object created", Singleton.getInstance()) ## Gets created here
s1 = Singleton() ## instance already created


class Pizza(object):
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    @staticmethod
    def compute_area(radius):
         import math
         return math.pi * (radius ** 2)

    @classmethod
    def compute_volume(cls, height, radius):
         return height * cls.compute_area(radius)

    def get_volume(self):
        return self.compute_volume(self.height, self.radius)

p = Pizza(10,2)
print ("Calculate Pizza area: {0}".format(p.compute_area(5)))
print ("Calculate Pizza volume: {0}".format(Pizza.compute_volume(10, 1.5)))
print ("Get Pizza volume: {0}".format(p.get_volume()))


