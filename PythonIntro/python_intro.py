# python_intro.py
"""Python Essentials: Introduction to Python.
<Name> Jean Philip L. Juachon
<Class> Python Lab 1
<Date>July 16, 2019
"""
#PROBLEM 1
#PROBLEM 2
def sphere_volume(r): #creation of the sphere_volume function where r is the input
    ab = 3.14159 #setting the default pi variable
    return 4/3*ab*r**3 #returning the formula for sphere_volume 

#PROBLEM 3
def isolate(a,b,c,d,e):
    print (a,b,c, sep = "     ", end = " ")#set the sep to 5 spaces and end it with 1 space.
    print (d, e) #print the last two numbers, no "sep" argument included because the default space is one.

#PROBLEM 4
def first_half(string1):
    lengthA = len(string1) #getting the length of string, then store it in variable lengthA
    splitA = round(lengthA/2) #getting the half count of the word, "round" to remove the middle letter if it has a remainder
    halfWord = string1[:splitA] #pass the string to var"halfWord", and get only the half of it
    return halfWord
def backward(string2):
    backwardString = string2[::-1] #pass the string to var"backwardString" with [::-1] to reverse it
    return backwardString

#PROBLEM 5
def list_ops():
    animals = ["bear", "ant", "cat", "dog"] #store the list of animals
    animals.append("eagle") #append or add eagle on the last part of the list
    animals.insert(2, "fox") #insert fox on the 2nd index
    del animals[1] #delete the second animal on the list.
    animals.sort(reverse = True)#sort the animals reverse alphabetically, thus the statement reverse = True
    animals.insert(1,"hawk") #replace eagle with hawk
    del animals[2] #delete the third animal
    animals.append("hunter") #add hunter on the last part of the list
    del animals[3]#delete fox
    del animals[3]
    del animals[3]
    animals.append("bearhunter")
    return animals

#PROBLEM 6
def pig_latin(word):
    letter = word[0] #get the first letter of the written word then store it in a variable called "letter"
    if letter == "a": #check the stored letter if it is vowel or consonant
        return (word +"hay") #print the word plus the given condition(if vowel or consonant)....
    elif letter =="e":
        return (word +"hay")
    elif letter == "i":
        return (word +"hay")
    elif letter == "o":
        return (word +"hay")
    elif letter == "u":
        return (word +"hay")
    else:
        return (word[1:]+letter+ "ay") #condition if letter = consonant. get the set of letters after 1, then add the letter on the last, then add "ay"

#PROBLEM 7
def palindromeChecker(numberA):
    return str(numberA) == str(numberA)[::-1] ##get the reverse of the input, convert to string
def palindrome(): #accepts inputs of two numbers
    z = 0 #initialize z to 0
    for num1 in range(999,99,-1):
        for num2 in range(999,99,-1):
            if palindromeChecker(num1*num2):
                if num1*num2 > z:
                    z = num1*num2
    return z

#PROBLEM 8
def alt_harmonic(integerA):
    list_trial = [(-1)**(i+1)/i for i in range(1,integerA+1)] #formula to initialize 
    return (sum(list_trial))
    
    
if __name__ == "__main__":
    print("Hello World!") #print hello world
    print("The volume of the sphere with radius = 4 is:",sphere_volume(4)) #print the sphere_volume where r = 4
    isolate(1,2,3,4,5) #print sample values 1,2,3,4,5
    print(first_half("kelseymerritt"))
    print(backward("argument"))
    print(list_ops())
    pig_latin("apple")
    pig_latin("thermal")
    palindromeChecker(956011)
    print(alt_harmonic(5000))





