#!/usr/bin/env python
# coding: utf-8


# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>Jean Philip L. Juachon
<Class>
<Date>July 5, 2019
"""

from random import choice


# Problem 1
'''
Problem 1 has four steps, for step 1, it raises a value error if the input is
not a 3 digit number,  and if the difference of the first and last digit is less than 1.
Step 2 asks for the reverse of the input then raises an error if it is not the reverse
Step 3 raises an error if the entered number is not the positive difference
step 4 raises and error if the entered number is not the reverse of step 3's number
'''
def arithmagic():
    step_1 = int(input("Enter a 3-digit number where the first and last digits differ by 2 or more: "))
    if step_1 < 100 or step_1 >= 1000:
        raise ValueError('Please input a 3 digit number')
    firstDigit = step_1 / 100
    lastDigit = step_1 % 10
    difference = firstDigit - lastDigit
    if abs(difference) < 1:
        raise ValueError("Please make sure that the difference is 2 or more")
        
    step_2 = input("Enter the reverse of the first number, obtained "
                                          "by reading it backwards: ")
    if step_2 != str(step_1)[::-1]:
        raise ValueError("Please input the reverse of your first number")
        
    step_3 = int(input("Enter the positive difference of these numbers: "))
    difference2 = abs(step_1 - int(step_2))
    if step_3 != difference2:
        raise ValueError("Please input the difference of the first 2 numbers")
        
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4 != str(step_3)[::-1]:
        raise ValueError("Please input the reverse of your third number")
    else:
        print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
        """
        the function raises an error and stops the iteration when ctrl+c is pressed
        """
def random_walk(max_iters=1e12):
    walk = 0
    directions = [1, -1]
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
         print("Interrupted iteration: ",i)
    return walk
        

# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter():
    def __init__(self, fileName):
        try:
            with open(fileName, 'r') as i:
                self.fileName = fileName
                self.Contents = i.read()
                self.total = len(self.Contents)
                self.alphabetic = 0
                self.numerical = 0
                lines = self.Contents.split("\n")
                self.whitespace = len(lines)
                self.lines = len(lines)
                for i in range(len(lines)):
                    for j in range(len(lines[i])):
                        if lines[i][j].isalpha():
                            self.alphabetic +=1
                        elif lines[i][j].isdigit():
                            self.numeric +=1
                        elif lines[i][j].isspace():
                            self.whitespace +=1
#                fileName.close()
                            
                
        except (FileNotFoundError, TypeError, OSError):
            newFileName = input("Please enter a valid file name: ")
            filterObject = ContentFilter(newFileName)
            self.fileName = filterObject.fileName
            self.Contents = filterObject.Contents
            self.total = filterObject.total
            self.alphabetic = filterObject.alphabetic
            self.numerical = filterObject.numerical
            self.whitespace = filterObject.whitespace
            self.lines = filterObject.lines
            
            
    def uniform(self, fileName, mode = 'w', case = "upper"):
        if mode not in ('w','x','a'):
            raise ValueError("Please enter the right attribute mode: 'w','x','a'")
        if case != "upper" and case != "lower":
            raise ValueError("blablabla")
        with open(fileName, mode) as varOutput:
            if case == "upper":
                varOutput.write(self.Contents.upper())
            else:
                varOutput.write(self.Contents.lower())
                
    def reverse(self, fileName, mode = 'w', unit = "line"):
        if mode not in ('w','x','a'):
            raise ValueError("Please enter a valid mode: 'x', 'w','a' ")
        if unit not in ('line','word'):
            raise ValueError("Attribute should either be 'line' or 'word'")
        with open(fileName,mode) as varOutput2:
            lines = self.Contents.strip().split("\n")
            if unit == "line":
                for i in range(len(lines)-1,-1,-1):
                    varOutput2.write(lines[i]+"\n")
            else:
                for i in range(len(lines)):
                    words = lines[i].split(" ")
                    for j in range(len(words)-1,-1,-1):
                        varOutput2.write(words[j]+ " ")
                    varOutput2.write("\n")
    
    def __str__(self):
        src = "Source file:\t\t\t"+self.fileName+"\n"
        total = "Total Char:\t\t\t" +str(self.total)+"\n"
        alpha = "Alphabetical characters:\t"+str(self.alphabetic)+ "\n"
        num = "Numerical characters:\t\t" +str(self.numerical) +"\n"
        space = "Whitespace:\t\t\t" +str(self.whitespace) +"\n"
        lines = "Number of lines:\t\t" +str(self.lines)
        return src+total+alpha+num+space+lines                
                        
                #    
                    
   # def transpose(self)



