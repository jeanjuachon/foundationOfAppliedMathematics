# -*- coding: utf-8 -*-
# standard_library.py
"""Python Essentials: The Standard Library.
<Name> Jean Philip L Juachon
<Class> BUDS 2019
<Date> July 17, 2019
"""
import box
import time
import random
import sys

#PROBLEM 1
def prob1(L): #function that accepts an input list
    return min(L), max(L), sum(L)/len(L) #minimum, maximum. average of the input list

#PROBLEM 2
def prob2():
    '''
    Tests to check whether mutable or immutable
    '''
    int1 = 9
    int2 = int1
    int2 = 6
    int2 == int1
    if int2 == int1:
        print("Integer is mutable: ")
    else:
        print("Integer is immutable: ")
 
    string1 = "hello"
    string2 = string1
    string2 = "hola"
    string2 == string1
    if string1 == string2:
        print("String is mutable: ")
    else:
        print("String is immutable: ")

    list1 = [1,2,3,4,5]
    list2 = list1
    list2[0] = 0
    list2 == list1
    if list1 == list2:
        print("List is mutable: ")
    else:
        print("List is immutable: ")

    tuple1 = ("bag",1,"rose")
    tuple2 = tuple1
    tuple2 = ("watch",7,"head")
    tuple2 == tuple1
    if tuple1 == tuple2:
        print("Tuple is mutable: ")
    else:
        print("Tuple is immutable: ")

    set1 = {1,2,3,4,5}
    set2 = set1
    set2 = {5,6,7,8}
    set2 == set1
    if set1 == set2:
        print("Set is mutable: ")
    else:
        print("Set is immutable: ")
    
#PROBLEM 3
'''
create a function that accepts 2 numbers
create calculator.py, import it with math, then use it to solve for the hypotenuse.
'''
def hypot(legA,legB):
    import calculator
    hypotenuse = calculator.squareRoot(calculator.addition(legA**2,legB**2))
    return hypotenuse

#PROBLEM 4
'''create the powerset of the inputs '''
from itertools import combinations #import chain and combination from itertools
def power_set(iterSet): #function that takes a set
    listA = []
    for i in range(len(iterSet)+1):
        listA += list(combinations(iterSet, i))
    for i in range(len(listA)):
        listA[i] = set(listA[i])
    return listA
   # setA = iterSet #set the input to variable setA
    #return chain.from_iterable(combinations(setA, element) for element in range(len(setA)+1))#use the imported chain a

#PROBLEM 5
def shutBox(name, timeInput):
    remaining_numbers = [1,2,3,4,5,6,7,8,9]
    rollTemp = 0
    elapsed_time = 0
    
    if sum(remaining_numbers) <= 6:
        rollTemp = random.randint(1,6)
    else:
        rollTemp = random.randint(2,12)
    print("Numbers left: ",remaining_numbers)
    print("Roll = ",rollTemp)
    
    while timeInput - elapsed_time > 0 and box.isvalid(rollTemp,remaining_numbers) and remaining_numbers !=[]:
        print("Remaining time: ", timeInput - elapsed_time)
        start_time = time.time()
        print("What choice to eliminate?: ")
        choices = input()
        end_time = time.time()
        elapsed_time += end_time - start_time
        eliminator = box.parse_input(choices, remaining_numbers)
        if sum(eliminator) == rollTemp:
            for i in range(len(eliminator)):
                remaining_numbers.remove(eliminator[i])
            if sum(remaining_numbers) <= 6:
                rollTemp = random.randint(1,6)
            else:
                rollTemp = random.randint(2,12)
            if elapsed_time < timeInput:
                print("Numbers left: ",remaining_numbers)
                print("Roll = ",rollTemp)
        else:
            print("Invalid input!")
            continue
    print("Score: ", sum(remaining_numbers))
    print("Time Played: ",round(elapsed_time),2)
    if remaining_numbers == []:
        print("congratulations")
    else:
        print("better luck next time!")
         
if len(sys.argv) == 3:
    shutBox(sys.argv[1], float(sys.argv[2]))



#if __name__ == "__main__":
#print(list_mma([1,2,3,4,5,6,6,6,6,6]))
#print(triangle(5,6))
#for setResults in power_set([3,6,9]):
 #  print(setResults)
