#!/usr/bin/env python3
# Assignment 1 Part 2
# Statistical NLP 1
# Author: Shadasha Williams

import sys
from collections import Counter
from itertools import islice
import numpy as np
from math import log


# splits the content of a file from specifies in the command line
file = sys.argv[1]
fn = open(file)
text = fn.read()
words = text.split()
# saves the text as a list
testData = words[-20000:]
after_words = words[:-20000]
# makes a list of the last N elements and returns a list of all of the elements minus the last N elements
heldoutData = after_words[-40000:]
trainingData = after_words[:-40000]
trainingDataText = "\n".join(trainingData)
testDataText = "\n".join(testData)
heldoutDataText = "\n".join(heldoutData)

# set up the training data for bi/uni/trigram counts
# get the size of the entire text
T = len(trainingData)
# get the vocabulary size from unigrams
trainingData = ["<S>", "<S>"] + trainingData
Unigrams = Counter(trainingData)
V = len(Unigrams)

bigramtext = trainingDataText
Bigramcounts = bigramtext.split("\n")

# creates a list of bigram counts
Bigrams = Counter(zip(Bigramcounts, islice(Bigramcounts, 1, None)))

trigramCounts = trainingData

# creates trigrams and counter
Trigrams = Counter(zip(trigramCounts, islice(trigramCounts, 1, None), islice(trigramCounts, 2, None)))

# compute uniform probability
P0 = 1 / V
# compute trigram probability
for u in Trigrams:
    if (u[0], u[1]) in Bigrams:
        Trigrams[u] = Trigrams[u] / Bigrams[u[0], u[1]]
# compute bigram probability
for b in Bigrams:
    if b[0] in Unigrams:
        Bigrams[b] = Bigrams[b] / Unigrams[b[0]]

# compute unigram probability
for k in Unigrams:
    Unigrams[k] = Unigrams[k] / T

P3 = Trigrams
P2 = Bigrams
P1 = Unigrams


# returns the results of the smoothing using EM algorithm and set epsilon for convergence of lambdas
def em_alg(data):
    eps = 0.0001
    # start initial lambdas with random values such that they are all non-negative and sum up to one
    intlambdas = np.random.random(4)
    intlambdas /= intlambdas.sum()
    # compute trigrams from input data with special symbols added
    data = "<S>\n<S>\n" + data
    data = data.split()
    trigrams = list(zip(data, islice(data, 1, None), islice(data, 2, None)))
    # set the following values to arrays of len 4 with 0 as the values
    exp_counts = np.zeros(4)
    lambda_next = np.zeros(4)
    lambdas = intlambdas
    # compute expected counts for all j in lambda from summing the values of the lambda to the probability
    # create the prime value to be used in expected count
    # set lambdas to intlambdas and iterate while the difference of lambdas
    # and next lambdas j is greater than the epsilon
    while True:
        for x in trigrams:
            p_prime = lambdas[3] * P3[x] + lambdas[2] * P2[x[1], x[2]] + lambdas[1] * P1[x[2]] + lambdas[0] * P0
            exp_counts[0] += lambdas[0] * P0 / p_prime
            exp_counts[1] += lambdas[1] * P1[x[2]] / p_prime
            exp_counts[2] += lambdas[2] * P2[x[1], x[2]] / p_prime
            exp_counts[3] += lambdas[3] * P3[x] / p_prime
            # computes a new set of lambdas using the next function
        for i in range(len(exp_counts)):
            lambda_next[i] = exp_counts[i] / sum(exp_counts)

        if all(l < eps for l in lambdas - lambda_next):
            break
        exp_counts = np.zeros(4)
        lambdas = np.copy(lambda_next)

    return lambdas

print("Smoothing lambdas from the held out data are ")
print(em_alg(heldoutDataText))
# print out the parameters when it is run on the training data
print("Smoothing lambdas from the training data are ")
print(em_alg(trainingDataText))


def cross_entropy(lambdas):
    # takes the length of the test data as T' with added symbols for appropriate trigram counts
    data = ["<S>", "<S>"] + testData
    t_prime = len(data)
    # turns data in to trigrams used for the formula
    trigrams = (zip(data, islice(data, 1, None), islice(data, 2, None)))
    # create variable to loop over all n from 1 to T'
    prob_sum = 0
    # use the smoothed model for  all of the trigrams in the test data
    for x in trigrams:
        p_prime = lambdas[3] * P3[x] + lambdas[2] * P2[x[1], x[2]] + lambdas[1] * P1[x[2]] + lambdas[0] * P0
        prob_sum += log(p_prime, 2)
    cross_ent = (-1 / t_prime) * prob_sum
    print(cross_ent)
    return cross_ent

# run the cross entropy using the lambdas computed from the heldout data
testLambdas = em_alg(heldoutDataText)
print("Cross entropy for the text" + file)
cross_entropy(testLambdas)


# tweaking the smoothing parameters by increasing the trigram
def lambdas_tweaking_increase(lambdas, percent):
    # load two arrays for manipulation
    diff = 1 - lambdas[3]
    new_lambdas = ["0", "0", "0", 0]
    # creates a list to add the final lambdas to
    final_lambdas = []
    # calculates x amount of difference
    for p in range(len(percent)):
        perc_diff = diff * percent[p]
        new_lambdas[3] = lambdas[3] + perc_diff
        # changes the trigrams parameter by specified percentage
        # sets sum of all of the other values in the new lambdas
        n = 1 - new_lambdas[3]
        # get three random float numbers divide them by there sum and multiply by
        # n for normal distribution
        randoms = np.random.random(3)
        for y in randoms:
            new_lambdas.insert(0, y/randoms.sum() * n)
        # remove the zeros in the list
        while "0" in new_lambdas:
            new_lambdas.remove("0")
        final_lambdas.append(new_lambdas)
        new_lambdas = ["0", "0", "0", 0]
    return final_lambdas


# tweaking the parameters by decreasing the trigram
def lambdas_tweaking_decrease(lambdas, percent):
    # loads the two arrays
    new_lambdas = ["0", "0", "0", 0]
    # creates new list for list of altered lambdas
    final_lambdas = []
    for r in range(len(percent)):
        new_lambdas[3] = lambdas[3] * percent[r]
        n = 1 - new_lambdas[3]
        # gets the difference for sum
        randoms = np.random.random(3)
        for y in randoms:
            new_lambdas.insert(0, y/randoms.sum() * n)
        # remove the zeros in the list
        while "0" in new_lambdas:
            new_lambdas.remove("0")
        final_lambdas.append(new_lambdas)
        new_lambdas = ["0", "0", "0", 0]
    return final_lambdas

# list of percentages to be used for cross entropy recalculation changed 0 to small number to normalize
first_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
second_percent = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

increasing_list = lambdas_tweaking_increase(testLambdas, first_percent)
decreasing_list = lambdas_tweaking_decrease(testLambdas, second_percent)
# compute cross entropy for all of the new parameters from tweaking
print("cross entropy for increased tweaks")
for inc in increasing_list:
    print(inc)
    cross_entropy(inc)

print("cross entropy for decreased tweaks")
for dec in decreasing_list:
    print(dec)
    cross_entropy(dec)

# compute the graph coverage for percentage of test data not seen in
# training data

# get a list of all of the unique words so that there are not multiple counts in sum
uniq_counts = []
for uniq in testData:
    if uniq not in uniq_counts:
        uniq_counts.append(uniq)
# sum all of the words not seen in training data over the vocabulary of the test data
sum_counts = 0
for word in uniq_counts:
    if P1[word] > 0:
        sum_counts += 1
coverage = sum_counts / len(uniq_counts)
print("The graph coverage for this file is " + str(coverage * 100))
