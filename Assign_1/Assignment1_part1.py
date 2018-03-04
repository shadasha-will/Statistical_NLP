#!/usr/bin/env python3
#Author: Shadasha Williams
#Statistical NLP 1 
#Assignment One Part One

from collections import Counter
from collections import defaultdict
from math import log
from decimal import Decimal
from itertools import islice
import random
import numpy as np

#computes the conditional entropy
def conditional_entropy(text):
    #splits the text into words and lines
    words = len(text.split())
    lines = text.split("\n")
    #creates dictionaries of unigrams from both of the texts
    Unigrams = defaultdict(int)
    for word in lines:
        Unigrams[word] += 1
    #creates bigram counter from the text
    Bigrams = Counter(zip(lines, islice(lines, 1, None)))
    #sets the frequency sum as a float and gets total bigram count
    freqSum = 0.0
    BigramSum = float(words - 1)

    #loops through the unigrams and bigrams for conditional and and joint probability
    for word in Bigrams:
        if word[0] in Unigrams:
            prob1 = float(Bigrams[word]) / BigramSum
            prob2 = float(Bigrams[word]) / Unigrams[word[0]]
            freqSum += prob1 * log(prob2, 2)
    cond_entropy = -freqSum
    return cond_entropy

#runs a text on the Czech and English texts to compute entropy and complexity
def entropy_test(file):
    #opens specified file and read it
    fn = open(file)
    text = fn.read()
    #runs the test on a given text
    test_ent = conditional_entropy(text)
    print("The entropy  of this text is", test_ent)
    perplexity = pow(2, test_ent)
    print("The perplexity of this text is", perplexity)

#runs the English text first then the Czech
print("For the English text:")
entropy_test('TEXTEN1.txt')
print("For the Czech text:")
entropy_test('TEXTCZ1UTF.txt')

#below in the method for the mess-up texts
#first we mess up the text by characters by assigning each character to a random variable and changing the
#character if it meets certain conditions specified

def char_mess_up(file, probability):
    #reads the original file and saves it as a list
    fr=open(file)
    text = fr.read()
    #creates a list of all of letters used in the file to be chosen from randomly later
    letters = list(text)
    uniqletters = []
    for x in letters:
        if x != "\n":
            if x not in uniqletters:
                uniqletters.append(x)
    #create a list of lists for all words in the text as well as characters
    entText = []
    for line in text.split():
        char = list(line)
        entText.append(char)
    #creates a values for each of the characters in the text using random
    for word in entText:
        for char in range(len(word)):
            randFloat = random.random()
            if randFloat < probability:
                word[char] = random.choice(uniqletters)
    #joins the words and letters to create a words on new lines
    lettersJoined = []
    for line in entText:
        for x in line:
            ''.join(str(x))
        lettersJoined.append("".join(line))
    newText = "\n".join(lettersJoined)
    print("Entropy of messed up characters of " + str(probability) + " of file " + file)
    mess_char_ent = conditional_entropy(newText)
    print(mess_char_ent)

#a function that messes up the words using the previous method for characters

def word_mess_up(file, probability):
    fn = open(file)
    text = fn.read()
    #creates a list of all unique words in the text to be chosen from randomly
    words = text.split()
    uniqWords = []
    for x in words:
        if x != "\n":
            if x not in uniqWords:
                uniqWords.append(x)
    #read the entire text and assign variable to input to be randomized
    for word in range(len(words)):
        randFloat = random.random()
        if randFloat < probability:
            words[word] = random.choice(uniqWords)
    newText =  "\n".join(words)
    mess_ent = conditional_entropy(newText)
    print("Entropy of mess up words of " + str(probability) + " of file " + file + " is ")
    print(mess_ent)

#a function that runs the messed up characters 10 times
def messed_up_text(file):
    for i in range(10):
        word_mess_up(file, .1)
        word_mess_up(file, .05)
        word_mess_up(file , .01)
        word_mess_up(file, .001)
        word_mess_up(file, .0001)
        word_mess_up(file, .00001)
    #messes up the characters as well as words and performs such under given probabilities
    #10 times so that we can get a number of random results to examine
        char_mess_up(file, .1)
        char_mess_up(file, .05)
        char_mess_up(file, .01)
        char_mess_up(file, .001)
        char_mess_up(file, .0001)
        char_mess_up(file, .00001)

messed_up_text('TEXTEN1.txt')
messed_up_text('TEXTCZ1UTF.txt')




