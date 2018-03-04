#! user/bin/env python3
# Author: Shadasha Williams
# Assignment 3
# Statistical NLP II

from collections import Counter
from collections import defaultdict
from itertools import islice
import sys
import numpy as np
from math import log


# load the file from the command line
file = sys.argv[1]
text = open(file, 'r')
lines = text.read()
words = lines.split("\n")
# do the original splitting
S_data = words[-40000:]
rest = words[:-40000]
H_data = rest[-20000:]
T_data = rest[:-20000]

# open the data and split into sentences
t = "\n".join(T_data)
sents = t.split("###/###")
# add start and end to each of the sentences
train = []
tag_list = []
word_list = []
all_seq = []
for sent in sents:
    if sent:
        # add two additional tags for trigrams
        new = [("<START>", "<START>"), ("<START>", "<START>")]
        words = sent.split("\n")
        for word in words:
            if word:
                # split the tokens into word and tag
                new.append(tuple(word.split("/")))
                tag_list.append(word.split("/")[1])
                word_list.append(word.split("/")[0])
                all_seq.append(tuple(word.split("/")))
        new.append(("<END>", "<END>"))
        tag_list.append("<END>")
        train.append(new)
# get the distribution of the tags
all_tags = set(tag_list)
T = len(tag_list)
V = len(all_tags)
unigrams_tags_counts = Counter(tag_list)
trigrams = []
bigrams = []
# get the bigram and trigram counts from each of the sentences in the training data
for s in train:
    if s:
        tags = []
        for word in s:
            tags = word[1]
            tag_list.append(tags)
        bigrams_sent = list(zip(tag_list, islice(tag_list, 1, None)))
        trigrams_sent = list(zip(tag_list, islice(tag_list, 1, None), islice(tag_list, 2, None)))
        for x in bigrams_sent:
            bigrams.append(x)
        for y in trigrams_sent:
            trigrams.append(y)
# get a counter of all of the trigrams and bigrams from the sents
trigram_tags = Counter(trigrams)
bigram_tags = Counter(bigrams)
# get the trigram values for parameters
for x in trigram_tags:
    trigram_tags[x] = trigram_tags[x] / bigram_tags[x[0], x[1]]
# get the bigram for paramaters
for s in bigram_tags:
    bigram_tags[s] = bigram_tags[s] / unigrams_tags_counts[s[0]]
# get the unigram counts
unigrams_tags = defaultdict(int)
for x in unigrams_tags_counts:
    unigrams_tags[x] = unigrams_tags_counts[x] / T
# get the uniform probability
uniform_tags = 1 / V
# get the word and tag values for parameters
all_seq_counts = Counter(all_seq)
word_tag = defaultdict(int)
for x in all_seq_counts:
    word_tag[x] = all_seq_counts[x] / unigrams_tags_counts[x[1]]

# start the EM smoothing
def em_smoothing(data):
    eps = 0.0001
    # start initial lambdas with random numes
    intlambdas = np.random.random(4)
    intlambdas /= intlambdas.sum()
    # split into sentences and get the trigrams from each sentence
    heldout = "\n".join(data)
    held_sents = heldout.split("###/###")
    trigrams = []
    for sent in held_sents:
        if sent:
            # add additional symbols for correct trigram counts
            new = [("<START>"), ("<START>")]
            words = sent.split("\n")
            for word in words:
                if word:
                    word_tag = tuple(word.split("/"))
                    new.append(word_tag[1])
            new.append(("<END>"))
            # add each trigram from the sentence to a list of all trigrams
            sent_tri = list(zip(new, islice(new, 1, None), islice(new, 2, None)))
            for x in sent_tri:
                trigrams.append(x)
    # set the following arrays to zero
    exp_counts = np.zeros(4)
    lambda_next = np.zeros(4)
    lambdas = intlambdas
    # compute expects counts for all j in lambda from summing the values of the lmada to the prob
    # create prime value to be used in expected counts
    while True:
        for x in trigrams:
            p_prime = lambdas[3] * trigram_tags[x] + lambdas[2] * bigram_tags[x[1], x[2]] \
                    + lambdas[1] * unigrams_tags[x[2]] + lambdas[0] * uniform_tags
            exp_counts[0] += lambdas[0] * uniform_tags / p_prime
            exp_counts[1] += lambdas[1] * unigrams_tags[x[2]] / p_prime
            exp_counts[2] += lambdas[2] * bigram_tags[x[1], x[2]] / p_prime
            exp_counts[3] += lambdas[3] * trigram_tags[x] / p_prime
            # compute a new set of lambdas
        for i in range(len(exp_counts)):
            lambda_next[i] = exp_counts[i] / sum(exp_counts)
        if all(l < eps for l in lambdas - lambda_next):
            break
        exp_counts = np.zeros(4)
        lambdas = np.copy(lambda_next)
    return lambdas

# get the lambdas and print them
lambdas_result = em_smoothing(H_data)
print("Lambdas after EM smoothing ", lambdas_result)


def get_emm_param(word,tag):
    # the input is a tuple of word and tag
    # gets the smoothed emmissions value
    emm_val = all_seq_counts[(word, tag)] + 1 / unigrams_tags_counts[tag]
    return emm_val

def get_trans_param(trigram):
    # get the estimate of the trigrams using the lambdas
    # uses linear interpolation
    trans_val = lambdas_result[3] * trigram_tags[trigram] \
              + lambdas_result[2] * bigram_tags[trigram[1], trigram[2]] \
              + lambdas_result[1] * unigrams_tags[trigram[2]] \
              + lambdas_result[0] * uniform_tags
    return trans_val

# start the viterbi algorithm for supervised learning
def viterbi_alg(data):
    # initialize by setting the values
    n = len(data)
    states = list(unigrams_tags.keys())
    # set of the alpha values
    viterbi = defaultdict(int)
    # initialize so that all sequences start with the start symbol
    viterbi[0,"<START>","<START>"] = 1
    # intialize the backpointer as a dictionary
    bp = {}
    bp["<START>", "<START>"] = []
    # start the recursive process
    for k in range(1, n + 1):
        for u in range(len(states) - 1):
            for v in range(len(states)):
                w = u - 1
                viterbi[k, states[u], states[v]] = max(viterbi[k-1, states[w], states[u]]  \
                                                 * get_trans_param((states[w], states[u], states[v])) \
                                                 * get_emm_param(data[k], states[u]))
                # get the forward and backwards values
                bp[k, states[u], states[v]] = viterbi[k-1, states[w], states[u]]  \
                                                 * get_trans_param((states[w], states[u], states[v])) \
                                                 * get_emm_param(data[k], states[u]))


sent = ["This", "is", "a", "baby", ".", "<END>"]
viterbi_alg(sent)














