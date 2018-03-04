#!usr/bin/env python3
# Author : Shadasha Williams
# Word Classes Hierarchy

from itertools import islice, combinations
from collections import Counter, defaultdict
from math import log2
import sys


class Wordclasses(object):
    def __init__(self, words, trainingdata):
        self.bigram_matrix = defaultdict(Counter)
        first = words[0]
        for second in words[1:]:
            self.bigram_matrix[first][second] += 1
            first = second
        # fill the dictionary with bigrams from the original 8000
        # gets all of the bigrams of the 8000
        self.bigrams = list(zip(words, islice(words, 1, None)))
        self.N = len(self.bigrams)

        # gets the left and right counts for all of the words
        self.c_kl = Counter([left for left, right in self.bigrams])
        self.c_kr = Counter([right for left, right in self.bigrams])

        # fill a dictionary with the q_k
        self.bigram_q_k = defaultdict()

        # get the pointwise mutual information for the initial numbers
        self.init_mi = 0
        for x1 in self.bigram_matrix:
            for x2 in self.bigram_matrix[x1]:
                self.bigram_q_k[(x1, x2)] = (self.mutual_inf(self.c_kl[x1], self.c_kr[x2], self.bigram_matrix[x1][x2]))
                self.init_mi += self.bigram_q_k[(x1, x2)]
        # print the intial mutual information
        print("The initial mutual information is ", self.init_mi)
        # each word is initially in its own class
        self.classes = trainingdata

        # get the intial s_k
        self.initial_s_k = defaultdict()
        for x in self.classes:
            self.initial_s_k[x[0]] = self.s_k(x[0])

        # create a list of all combinations of the classes
        self.class_samp = [x for x, y in self.classes]

        # get only the words
        self.class_comb = combinations(self.class_samp, 2)

        # create a dictionary of the losses
        self.L_k_table = defaultdict(int)
        for x, y in self.class_comb:
            self.L_k_table[x, y] = self.L_k(x, y)
            # make the table symmetric
            self.L_k_table[y, x] = self.L_k_table[x, y]

        # create a list of merges
        self.history = []

    def merge(self, x, y):
        # merges the classes
        merged_class = (x, y)

        # remove the sets and add the merged to the classes
        self.class_samp.remove(x)
        self.class_samp.remove(y)
        self.class_samp.append(merged_class)

        # get both of the trailing words
        x_y = self.bigram_matrix[x] + self.bigram_matrix[y]

        # fill a dictionary with neighbors for further steps
        neighbors_x_y = set()
        for k in x_y.keys():
            if x_y[k] > 0:
                neighbors_x_y.add(k)
        for k in self.bigram_matrix.keys():
            if self.bigram_matrix[k][x] > 0:
                neighbors_x_y.add(k)
            if self.bigram_matrix[k][y] > 0:
                neighbors_x_y.add(k)

        neighbors_x_y.discard(x)
        neighbors_x_y.discard(y)

        # update loss table
        # choose two random from the list of neighbors and update loss
        for i, j in combinations(neighbors_x_y, 2):
            self.L_k_table[i, j] -= self.initial_s_k.get(i, 0)
            self.L_k_table[i, j] -= self.initial_s_k.get(j, 0)
            # q(i+j,x)
            self.L_k_table[i, j] += self.mutual_inf(self.c_kl[i] + self.c_kl[j], self.c_kr[x],
                                                    self.bigram_matrix[i][x] + self.bigram_matrix[j][x])
            # q(x,i+j)
            self.L_k_table[i, j] += self.mutual_inf(self.c_kl[x], self.c_kr[i] + self.c_kr[j],
                                                    self.bigram_matrix[x][i] + self.bigram_matrix[x][j])
            # q(i+j,y)
            self.L_k_table[i, j] += self.mutual_inf(self.c_kl[i] + self.c_kl[j], self.c_kr[y],
                                                    self.bigram_matrix[i][y] + self.bigram_matrix[j][y])
            # q(y,i+j)
            self.L_k_table[i, j] += self.mutual_inf(self.c_kl[y], self.c_kr[i] + self.c_kr[j],
                                                    self.bigram_matrix[y][i] + self.bigram_matrix[y][j])
            # make sure there is symmetry in the loss matrix
            self.L_k_table[j, i] = self.L_k_table[i, j]

        # update bigrams and unigrams

        # get the left neighbors of the terms and add to the dictionary
        self.bigram_matrix.pop(x)
        self.bigram_matrix.pop(y)
        self.bigram_matrix[merged_class] = x_y

        # update the rows(nested dict values)
        for h in self.bigram_matrix.keys():
            if x in self.bigram_matrix[h].keys() or y in self.bigram_matrix[h].keys():
                self.bigram_matrix[h][merged_class] = self.bigram_matrix[h][x] + self.bigram_matrix[h][y]
                # and delete the previous values
                self.bigram_matrix[h].pop(x, None)
                self.bigram_matrix[h].pop(y, None)

        # update all the lefts & rights of new merged class
        self.c_kl[merged_class] = self.c_kl[x] + self.c_kl[y]

        self.c_kl.pop(x, None)
        self.c_kl.pop(y, None)
        self.c_kr[merged_class] = self.c_kr[x] + self.c_kr[y]
        self.c_kr.pop(x, None)
        self.c_kl.pop(y, None)

        # update the s_k values for the neighbor values
        for n in neighbors_x_y:
            if n != merged_class:
                self.initial_s_k[n] = self.initial_s_k.get(n, 0) - self.bigram_q_k.get((n, x), 0) \
                                      - self.bigram_q_k.get((x, n), 0) \
                                      - self.bigram_q_k.get((n, y), 0) - self.bigram_q_k.get((y, n), 0)

        # update all the bigram qualities for the neighboring words
        for s in neighbors_x_y:
            self.bigram_q_k[merged_class, s] = self.mutual_inf(self.c_kl[merged_class], self.c_kr[s],
                                                               self.bigram_matrix[merged_class][s])
        # we only need to update q for words adjacent to the merged class
        for s in neighbors_x_y:
            self.bigram_q_k[s, merged_class] = self.mutual_inf(self.c_kl[s], self.c_kr[merged_class],
                                                               self.bigram_matrix[s][merged_class])

        # update the s_k values
        for n in neighbors_x_y:
            self.initial_s_k[n] = self.initial_s_k.get(n, 0) + self.bigram_q_k.get((n, merged_class), 0) \
                                  + self.bigram_q_k.get((merged_class, n), 0)

        # update the merged class s_k value
        self.initial_s_k[merged_class] = self.s_k(merged_class)

        for i, j in combinations(neighbors_x_y, 2):
            if i != merged_class and j != merged_class:
                self.L_k_table[i, j] += self.initial_s_k.get(i, 0) + self.initial_s_k.get(j, 0)
                # q_k+1(merged class, i+j)
                self.L_k_table[i, j] -= self.mutual_inf(self.c_kl[i] + self.c_kl[j], self.c_kr[merged_class],
                                                        self.bigram_matrix[i][merged_class] + self.bigram_matrix[j][
                                                            merged_class])
                # q_k+1(i+j, merged class)
                self.L_k_table[i, j] -= self.mutual_inf(self.c_kl[merged_class], self.c_kr[i] + self.c_kr[j],
                                                        self.bigram_matrix[merged_class][i] +
                                                        self.bigram_matrix[merged_class][j])
                # make sure there is symmetry in the loss matrix
                self.L_k_table[j, i] = self.L_k_table[i, j]

        # now update L for the merged class
        # in the slides they say it needs to be done from the original formula
        for i in neighbors_x_y:
            self.L_k_table[merged_class, i] = self.L_k(merged_class, i)
            # symmetry
            self.L_k_table[i, merged_class] = self.L_k_table[merged_class, i]

    def get_classes(self, class_num):
        # iterate until we have the amount of classes that we want
        while len(self.class_samp) > class_num:
            # get the best pair for merging
            self.get_min_loss(self.class_comb)
            word_to_merge, loss = self.get_min_loss(self.class_samp)
            x, y = word_to_merge[0], word_to_merge[1]
            results = "Merging" + " " + str(x) + " " + "and" + " " + str(y) + " " \
                      "to new class " + "with the loss of " + str(loss)
            self.merge(x, y)
            self.history.append(results)

        return self.history

    # get the minimal classes
    def get_min_loss(self, class_set):
        word_to_keep = ''
        loss = float(10000)
        # choose a string and large number to get the minimum
        class_remain = combinations(self.class_samp, 2)
        for x, y in class_remain:
            if self.L_k_table[x, y] < loss and self.L_k_table[x, y] > 0:
                loss = self.L_k_table[x, y]
                word_to_keep = (x, y)
        return word_to_keep, loss

    # get the losses from the the add and subtract formulas
    def L_k(self, x, y):
        loss = self.sub_k(x, y) - (self.add_k(x, y) + self.mock_merge(x, y))
        self.L_k_table[x, y] = loss
        return loss

    # compute s_k
    def s_k(self, a):
        right_sum = 0
        left_sum = 0
        int_adj = 0
        for l in self.bigram_q_k:
            if a == l[0]:
                right_sum += self.bigram_q_k[l]
            if a == l[1]:
                left_sum += self.bigram_q_k[l]
        if (a, a) in self.bigram_q_k.keys():
            int_adj = self.bigram_q_k[a, a]
            if int_adj == 'None':
                int_adj = 0
        s_k_val = left_sum + right_sum - int_adj
        return s_k_val

    # compute the subtract from the losses
    def sub_k(self, a, b):
        if (a, b) in self.bigram_q_k:
            qk_ab = self.bigram_q_k[a, b]

        else:
            qk_ab = 0
        # check if they exist in the mutual information
        if (b, a) in self.bigram_q_k:
            qk_ba = self.bigram_q_k[b, a]
        else:
            qk_ba = 0

        sub_k_val = self.s_k(a) + self.s_k(b) - qk_ba - qk_ab
        return sub_k_val

    # compute the addition of the new merge for the losses table
    def add_k(self, a, b):
        sum_bi = 0
        # merge the dictionaries when word is on left
        a_b = self.bigram_matrix[a] + self.bigram_matrix[b]
        new_left = self.c_kl[a] + self.c_kl[b]
        for x in a_b:
            if x != a and x != b:
                xy = a_b[x] / self.N
                if xy != 0:
                    try:
                        sum_bi += self.mutual_inf(new_left, self.c_kr[x], a_b[x])
                    except:
                        print(x)
                        print(new_left, self.c_kr[x], a_b[x])

        # get the values when word is on the right
        new_right = self.c_kr[a] + self.c_kr[b]
        for x in self.bigram_matrix:
            xy = self.bigram_matrix[x][a] + self.bigram_matrix[x][b]
            if x != a and x != b:
                if xy != 0:
                    sum_bi += self.mutual_inf(self.c_kl[x], new_right,
                                              self.bigram_matrix[x][a] + self.bigram_matrix[x][b])

        return sum_bi

    # a function to get the final set of classes
    def get_final(self):
        return self.class_samp

    # get a mock merge for the a+b intersections
    def mock_merge(self, a, b):
        # prepares a mock merge for for loss tables
        new_right = self.c_kr[a] + self.c_kr[b]
        new_left = self.c_kl[a] + self.c_kl[b]

        # for the intersection just add both
        bigrams = self.bigram_matrix[a][a] \
                  + self.bigram_matrix[a][b] \
                  + self.bigram_matrix[b][b] \
                  + self.bigram_matrix[b][a]
        if bigrams == 0:
            comb = 0
            return comb
        else:
            comb = self.mutual_inf(new_right, new_left, bigrams)
            return comb

    def mutual_inf(self, first, second, bigrams):
        if bigrams != 0:
            q_k = bigrams / self.N * log2(self.N * bigrams / (first * second))
        else:
            return 0
        return q_k


# load the data
if sys.argv[1] == 'words':
    # check to
    def load(file):
        new_file = open(file)
        text = new_file.read()
        words = text.split("\n")
        words.insert(0, "<START>")
        words = words[:-1]
        # split the lines into words and tags
        for x in range(len(words)):
            words[x] = words[x].rsplit('/', 1)[0]
        # get the first 8000 words from the list
        # @@ 8001 to be very precise
        test_sample = words[:8001]
        counts = Counter(test_sample)
        # return only the words which occur more than 10 times
        new_sample = []
        for w in counts:
            if counts[w] >= 10:
                new_sample.append((w, counts[w]))
        test = Wordclasses(test_sample, new_sample)
        merges = test.get_classes(10)
        #for x in merges:
         #   sys.stdout.write(x + "\n")
        final_class = test.get_final()
        for x in final_class:
            sys.stdout.write(str(x) + "\n")


    #load("TEXTEN1.ptg.txt")
    #load("TEXTCZ1.ptg")

if sys.argv[1] == 'tags':
    def load(file):
        new_file = open(file)
        text = new_file.read()
        words = text.split("\n")
        words.insert(0, "<START>/<START>")
        words = words[:-1]
        # split the lines into words and tags
        for x in range(len(words)):
            words[x] = words[x].rsplit('/', 1)[1]
        # get the first 8000 words from the list
        test_sample = words
        counts = Counter(test_sample)
        # return only the words which occur more than 10 times
        new_sample = []
        for w in counts:
            if counts[w] >= 5:
                new_sample.append((w, counts[w]))
        test = Wordclasses(test_sample, new_sample)
        merges = test.get_classes(10)
        #for x in merges:
        #    sys.stdout.write(x + "\n")
        final_class = test.get_final()
        for x in final_class:
            sys.stdout.write(str(x) + "\n")
    load("TEXTEN1.ptg.txt")
    #load("TEXTCZ1.ptg")



