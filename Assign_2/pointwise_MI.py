#!usr/bin/env python3
# Author : Shadasha Williams
# Best Friends Assignment

# a function that reads the file and computes the pointwise mutual information
from collections import Counter
from itertools import islice
from math import log2
import sys

def mutual_inf(file):
    new_file = open(file)
    text = new_file.read()
    words = text.split()
    # get the size of the text
    text_size = len(words)
    # saves the words of the file as a list
    # get the individual counts and save them as a dictionary
    unigrams = Counter(words)
    # get the set of bigrams
    bigrams = zip(words, islice(words, 1, None))
    # get the size of all the bigrams
    bigram_counts = Counter(bigrams)
    bigram_size = sum(bigram_counts.values())
    # calculates the mutual information for the close words and saves them in a dictionary
    mutual_inf_bigrams = {}
    for count in bigram_counts:
        # for each of the bigrams split into two parts
        if count[0] in unigrams and unigrams[count[0]] >= 10:
            # makes sure that if the first word occurs less than ten times
            if count[1] in unigrams and unigrams[count[1]] > 10:
                # makes sure that the second words occurs at least once
                bigram_prob = bigram_counts[count] / bigram_size
                x_prob = unigrams[count[0]] / text_size
                y_prob = unigrams[count[1]] / text_size
                pointw_mi = log2(bigram_prob / (x_prob * y_prob))
                mutual_inf_bigrams[count] = pointw_mi
    sorted_mi = sorted(mutual_inf_bigrams.items(), key=lambda x:x[1], reverse= True)
    for x in sorted_mi:
        sys.stdout.write(str(x) + "\n")


def mutual_far(file):
    new_file = open(file)
    # calculates the mutual information for all words in 50 word window
    # for all words that are before
    # create a bag of words list
    text = new_file.read()
    words = text.split()
    N = len(words)
    unigrams = Counter(words)
    bow_list = []
    for i in range(len(words) - 51):
        # gets the number of words that are not in the last 50
        for j in range(i+2, i+51):
            # get the range of at least one word in between and at most 50 words after
            bow_list.append((words[i], words[j]))
    for i in range(len(words)-51, len(words)):
        # populate the last fifty words of the array
        for j in range(i+2, len(words)):
            bow_list.append((words[i], words[j]))
    bigram_counts = Counter(bow_list)
    bigram_size = sum(bigram_counts.values())
    mutual_inf_bigrams = {}
    for count in bigram_counts:
        # for each of the bigrams split into two parts
        if count[0] in unigrams and unigrams[count[0]] >= 10:
            # makes sure that if the first word occurs less than ten times
            if count[1] in unigrams and unigrams[count[1]] > 10:
                # makes sure that the second words occurs at least once
                bigram_prob = bigram_counts[count] / bigram_size
                x_prob = unigrams[count[0]] / N
                y_prob = unigrams[count[1]] / N
                pointw_mi = log2(bigram_prob / (x_prob * y_prob))
                mutual_inf_bigrams[count] = pointw_mi
    sorted_mi = sorted(mutual_inf_bigrams.items(), key=lambda x: x[1], reverse=True)
    for x in sorted_mi:
        sys.stdout.write(str(x) + "\n")


mutual_inf("TEXTCZ1UTF.txt")
mutual_inf("TEXTEN1.txt")
mutual_far("TEXTEN1.txt")
mutual_far("TEXTCZ1UTF.txt")