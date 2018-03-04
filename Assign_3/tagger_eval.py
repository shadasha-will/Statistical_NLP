#! user/bin/env python3
# Author: Shadasha Williams
# Assignment 3
# Statistical NLP II
from nltk.tag import CRFTagger
import itertools
import sys

def eval(test_tags, real_tags):
    # edit the sentences to get only the tags
    for x in range(len(test_tags)):
        for word in range(len(test_tags[x])):
            test_tags[x][word] = test_tags[x][word][1]
    # create flat lists from both data sets
    test_list = list(itertools.chain.from_iterable(test_tags))
    real_list = list(itertools.chain.from_iterable(real_tags))
    # create a list of tuples of the correct and assigned tag
    tags = list(zip(real_list, test_list))
    total = len(list(tags))
    correct = 0
    for x in tags:
        if x[0] == x[1]:
            correct += 1
    accuracy = correct / total
    return accuracy


def load(training, testing):
    ct = CRFTagger()
    # split the training into sentences
    t = "\n".join(training)
    sents = t.split("###/###")
    # split the sentences into tokens
    train = []
    for sent in sents:
        if sent:
            new = []
            words = sent.split("\n")
            for word in words:
                if word:
                    # split the tokens into word and tag
                    new.append(tuple(word.split("/")))
            train.append(new)
    # remove any blank sentences that have been added
    for t in train:
        if not t:
            train.remove(t)
    ct.train(train, 'model.crf.tagger')
    # test on the testing data
    s = "\n".join(testing)
    s_sents = s.split("###/###")
    test = []
    sent_tags = []
    for t in s_sents:
        if t:
            new = []
            right_tags = []
            words = t.split("\n")
            for word in words:
                if word:
                    # split the tokens into just words
                    new.append(word.split("/")[0])
                    # save the tags in a list to be used later
                    right_tags.append(word.split("/")[1])
            sent_tags.append(right_tags)
            test.append(new)
    tags = ct.tag_sents(test)
    return tags, sent_tags



# open the english file ans do appropriate splitting
def run_tagger(file):
    text = open(file, 'r')
    lines = text.read()
    words = lines.split("\n")
    # do the original splitting
    S_data = words[-40000:]
    rest = words[:-40000]
    H_data = rest[-20000:]
    T_data = rest[:-20000]
    accuracy = []
    #test1 = load(S_data, T_data)[0]
    #act1 = load(S_data, T_data)[1]
    #accuracy.append(eval(test1, act1))
    # get the first 40000 words instead of last
    S_prime_1 = words[:40000][:-1]
    rest_prime_1 = words[40000:]
    #T_prime = rest_prime_1[-20000:]
    #test2 = load(S_prime_1, T_prime)[0]
    #act2 = load(S_prime_1, T_prime)[1]
    #accuracy.append(eval(test2, act2))
    # get another data set from different splits
    S_prime_2 = words[-40000:]
    rest_prime_2 = words[:-40000]
    T_prime_1 = rest_prime_2[20000:]
    #test3 = load(S_prime_2, T_prime_1)[0]
    #act3 = load(S_prime_2, T_prime_1)[1]
    #accuracy.append(eval(test3, act3))
    # use some of the previous methods for splitting
    T_prime_2 = H_data[:20000]
    #test4 = load(S_data, T_prime_2)[0]
    #act4 = load(S_data, T_prime_2)[1]
    #accuracy.append(eval(test4, act4))
    # split the last way
    T_prime_3 = rest_prime_1[:20000]
    test5 = load(S_prime_1, T_prime_3)[0]
    act5 = load(S_prime_1, T_prime_3)[1]
    accuracy.append(eval(test5, act5))
    #print(accuracy)
run_tagger("textcz2.ptg")





    