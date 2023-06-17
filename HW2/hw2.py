import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=[0]*26
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        # Traverse each line in the file
        for line in f:
            letter = [x for x in line] #Split the line into characters
            #Tranverse through each letter in the line
            for l in letter:
                #Check if it's A-Z or a-z
                if((l >= 'A') and (l <= 'Z')):
                    X[ord(l)-ord('A')] += 1 #increment by 1
                elif((l >= 'a') and (l <= 'z')):
                    X[ord(l)-ord('a')] += 1 #increment by 1

    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
def summation(lang, X):
    sum = 0
    for i in range(len(X)):
        sum += X[i]*math.log(lang[i])
    
    return sum

def F_English(e, X):
    return math.log(0.6)+summation(e, X)

def F_Spanish(s,X):
    return math.log(0.4)+summation(s,X)

def compute_english(e, s, X):
    f_diff = F_Spanish(s,X) - F_English(e,X)
    if(f_diff >= 100):
        return 0
    elif(f_diff <= -100):
        return 1
    else:
        return 1/(1+math.exp(f_diff))

if __name__ == "__main__":
    print('Q1') #Print all 26 letters and how many times they occur
    X = shred('letter.txt') #Q1, calls shred function
    l = ord('A') #Make "l" be the unicode for A
    #For loop to print out each letter and how many times it occurs
    for i in X:
        print(f'{chr(l)} {i}')
        l += 1

    print('Q2') #Compute X1 * loge1/s1 and round it to 4 decimal points
    e,s = get_parameter_vectors() #Get the probabilities of each letter in english and spanish
    print('%.4f' % (X[0]*math.log(e[0]))) #Print X1*loge1
    print('%.4f' % (X[0]*math.log(s[0]))) #Print X1*logs1

    print('Q3') #Compute F(English/Spanish) and round it to 4 decimal points
    f_eng = round(F_English(e, X), 4) #Get F(English)
    f_spa = round(F_Spanish(s,X), 4) #Get F(Spanish)
    print('%.4f' % f_eng) #Print F(English)
    print('%.4f' % f_spa) #Print F(Spanish)
    
    print('Q4') #Compute P(Y = English | X) and round it to 4 decimals
    print('%.4f' % compute_english(e, s, X)) # Print P(Y = English | X) and round it to 4 decimals
