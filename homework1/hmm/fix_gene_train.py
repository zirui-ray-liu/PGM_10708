#! /usr/bin/python

__author__ = "Daniel Bauer <bauer@cs.columbia.edu>"
__date__ = "$Sep 12, 2011"

import sys
from collections import defaultdict
import math
import re
"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""


def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line:  # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            # phrase_tag = fields[-2] #Unused
            # pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word
        else:  # Empty line
            yield None
        l = corpus_file.readline()

class Hmm(object):
    """
    Stores counts for n-grams and emissions.
    """

    def __init__(self):
        self.emission_counts = defaultdict(int)

    def write_fixed_train(self, original_trainfile, output):
        # First write counts for emissions
        numeric_p = re.compile(r"^.*[0-9]+.*$")
        allcap_p = re.compile(r"^[A-Z]+$")
        lastcap_p = re.compile(r"^.*[A-Z]$")
        l = original_trainfile.readline()
        while l:
            line = l.strip()
            fields = line.split(" ")
            # phrase_tag = fields[-2] #Unused
            # pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            if line != '':
                n = self.emission_counts[word]
                if n < 5 and numeric_p.match(word):
                    word = "_NUMERIC_"
                elif n < 5 and allcap_p.match(word):
                    word = "_ALLCAP_"
                elif n < 5 and lastcap_p.match(word):
                    word = "_LASTCAP_"
                elif n < 5:
                    word = "_RARE_"
                _l = word + " " + fields[-1] + "\n"
            else:
                _l = "\n"
            output.write(_l)
            l = original_trainfile.readline()




if __name__ == "__main__":
    # Initialize a trigram counter
    counter = Hmm()
    # Collect counts
    f_train = file('./gene.train', 'r')
    corpus_iterator = simple_conll_corpus_iterator(f_train)
    for l in corpus_iterator:
        if l != None:
            counter.emission_counts[l] += 1
    f_train.close()
    f_train = file('./gene.train', 'r')
    f_fixed_train = file('./fixed_gene.train', 'w')
    counter.write_fixed_train(f_train, f_fixed_train)