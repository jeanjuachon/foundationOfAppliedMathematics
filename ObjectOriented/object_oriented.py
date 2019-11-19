# -*- coding: utf-8 -*-
"""Python Essentials: The Standard Library.
<Name> Jean Philip L Juachon
<Class> BUDS 2019
<Date> July 23, 2019
"""
import math
'''creation of backpack'''
class Backpack:
    
    '''constructor, takes in 3 inputs'''
    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size
        
    '''add items to the bag, prints no more room if no. of items > 5, then doesn't add the item'''
    def put(self, item): #method to put an item in Backpack
        if len(self.contents) == self.max_size:
            print("No more room!")           
        else:
            self.contents.append(item)  
            
    def dump(self): #clear the contents
        return self.contents.clear()
    
    '''function to test whether it is the same bag'''
    # Problem 3: Write __eq__() and __str__().
    def __eq__(self, other):
        if (len(self.contents) == len(other.contents) and self.name == other.name
            and self.color == other.color):
            return True
        else:
            return False
        
    '''outputs the details of the bag'''
    def __str__(self):
        owner = "Owner: \t"+self.name+"\n"
        color = "Color: \t"+self.color+"\n"
        size = "Size: "+str(len(self.contents))+"\n"
        max_size = "Max_size: "+str(self.max_size)+"\n"
        contents = "Contents: " +str(self.contents)+"\n"
        return owner + color + size +max_size+contents
    
'''function for testing'''                   
def test_backpack():
    testpack = Backpack("Barry", "black") # Instantiate the object.
    if testpack.name != "Barry": # Test an attribute.
        print("Backpack.name assigned incorrectly")
    for item in ["pencil", "pen", "paper", "computer", "mouse","keyboard"]:
        testpack.put(item) # Test a method.
    print("Contents:", testpack.contents)
    
# An example of inheritance. You are not required to modify this class.  
class Knapsack(Backpack):
    def __init__(self, name, color, max_size = 3):
        Backpack.__init__(self, name, color, max_size)
        self.closed = True
     
    def put(self, item): #function to put an item in the object 
        if self.closed:
            print("Im closed!")
        else:
            Backpack.put(self, item)
            
    """If the knapsack is untied, use the Backpack.take() method."""    
    def take(self, item):
        if self.closed:
            print("im closed!")
        else:
            Backpack.take(self, item)
            
    '''get the weight'''            
    def weight(self):
        return sum([len(str(item)) for item in self.contents])
    
'''creation of jetpack with inheritance from Backpack'''    
# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    def __init__(self, name, color, max_size = 2, fuel = 10): #new arguments, change maxsize to 2 and add fuel
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel #set fuel here since it is new, not inherited
        
    def fly(self, amount):
        self.amount = amount
        if self.amount > self.fuel:
            print("Not enough fuel!")
        else:
            return self.fuel - self.amount
            
    def dump(self):
        self.fuel = 0
        return self.fuel

# Problem 4: Write a 'ComplexNumber' class.
'''creation of complexNumber from scratch'''    
class ComplexNumber(object):
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
    '''conjugate'''
    def conjugate(self):
        return ComplexNumber(self.real, -self.imag) #set the complex of real and imag to c
    
    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)
    
    def __str__(self):
        if self.imag >= 0:
            return '(' + str(self.real) + "+" +str(self.imag)+'j)' 
        else:
            return '(' + str(self.real) +str(self.imag)+'j)'
        
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other):
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other):
        return ComplexNumber(self.real*other.real - self.imag*other.imag, self.imag*other.real + self.real*other.imag)
    
    def __truediv__(self,other):
        return ComplexNumber((self.real*other.real+self.imag*other.imag)/(other.real**2 + other.imag**2), 
                            (self.imag*other.real-self.real*other.imag)/(other.real**2 + other.imag**2))