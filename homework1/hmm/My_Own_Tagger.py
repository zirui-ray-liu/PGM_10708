#! /usr/bin/python

__author__ = "Daniel Bauer <bauer@cs.columbia.edu>"
__date__ = "$Sep 12, 2011"

import copy
import sys
from collections import defaultdict
import math
import numpy as np
import re
"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

np.seterr(all='raise')
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
            ne_tag = fields[-1]
            # phrase_tag = fields[-2] #Unused
            # pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else:  # Empty line
            yield (None, None)
        l = corpus_file.readline()

def logadd(lp,lq):
    if lp == float("-inf") and lq == float("-inf"):
        return float("-inf")
    if lq <= lp:
        return lp + log_one_plus(np.exp(lq-lp))
    else:
        return lq + log_one_plus(np.exp(lp-lq))

def log_one_plus(z):
    """
    :param z: a number which > -1
    :return: log(1+z)
    """
    assert z>-1, "argument must > -1 !!!"
    if np.fabs(z) <= 1e-4:
        return (-0.5 * z + 1.0) * z
    else:
        return np.log(1 + z)

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = []  # Buffer for the current sentence
    for l in corpus_iterator:
        if l == (None, None):
            if current_sentence:  # Reached the end of a sentence
                yield current_sentence
                current_sentence = []  # Reset buffer
            else:  # Got empty input stream
                sys.stderr.write("WARNING: Got empty input file/stream.\n")
                raise StopIteration
        else:
            current_sentence.append(l)  # Add token to the buffer

    if current_sentence:  # If the last line was blank, we're done
        yield current_sentence  # Otherwise when there is no more token
        # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
        # Add boundary symbols to the sentence
        w_boundary = (n - 1) * [(None, "*")]
        w_boundary.extend(sent)
        w_boundary.append((None, "STOP"))
        # Then extract n-grams
        ngrams = (tuple(w_boundary[i:i + n]) for i in xrange(len(w_boundary) - n + 1))
        for n_gram in ngrams:  # Return one n-gram at a time
            yield n_gram



class Hmm(object):
    """
    Stores counts for n-grams and emissions.
    """

    def __init__(self, n=3):
        assert n >= 2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.Q = set()
        self.word_dict = {}
        self.Q_map = {}


    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            # Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram])  # retrieve only the tags
            for i in xrange(2, self.n + 1):  # Count NE-tag 2-grams..n-grams
                self.ngram_counts[i - 1][tagsonly[-i:]] += 1

            if ngram[-1][0] is not None:  # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1  # count 1-gram
                self.emission_counts[ngram[-1]] += 1  # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None:  # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

            for word, ne_tag in self.emission_counts:
                self.emission[(word, ne_tag)] = self.emission_counts[(word, ne_tag)] / self.ngram_counts[0][(ne_tag,)]



    def write_counts(self, output, printngrams=[1, 2, 3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))

        # Then write counts for all ngrams
        for n in printngrams:
            for ngram in self.ngram_counts[n - 1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" % (self.ngram_counts[n - 1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.Q = set()
        self.word_dict = {}
        self.Q_map = {}

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.Q.add(ne_tag)
                if not self.word_dict.has_key(word):
                    self.word_dict[word] = 0
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM", ""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n - 1][ngram] = count
        self.Q = sorted(list(self.Q))
        for i in xrange(len(self.Q)): self.Q_map[i] = self.Q[i]

    def ViterbiLog(self, sent):
        d1, d2, ncol = len(self.Q), len(self.Q), len(sent)
        numeric_p = re.compile(r"^.*[0-9]+.*$")
        allcap_p = re.compile(r"^[A-Z]+$")
        lastcap_p = re.compile(r"^.*[A-Z]$")
        if ncol == 1:
            word = sent[0][1]
            if not self.word_dict.has_key(word):
                if numeric_p.match(word):
                    word = "_NUMERIC_"
                elif allcap_p.match(word):
                    word  = "_ALLCAP_"
                elif lastcap_p.match(word):
                    word = "_LASTCAP_"
                else:
                    word = "_RARE_"
            tagged_for_word = max(self.Q, key=lambda x: self.emission_counts[(word, x)]/self.ngram_counts[0][(x,)])
            return tagged_for_word, 0
        else:
            mat = np.zeros(shape=(d1, d2, ncol), dtype=float)  # prob
            matTb = np.zeros(shape=(d1, d2, ncol), dtype=int)  # backtrac
            #=====================================================================
            """
            initialize the mat matrix  
            """
            for i in xrange(0,d1):
                for j in xrange(0,d2):
                    w1, w2 = sent[0][1], sent[1][1] # 1st word & 2nd word in the sentence
                    if not self.word_dict.has_key(w1):
                        if numeric_p.match(w1):
                            w1 = "_NUMERIC_"
                        elif allcap_p.match(w1):
                            w1 = "_ALLCAP_"
                        elif lastcap_p.match(w1):
                            w1 = "_LASTCAP_"
                        else:
                            w1 = "_RARE_"
                    if not self.word_dict.has_key(w2):
                        if numeric_p.match(w2):
                            w2 = "_NUMERIC_"
                        elif allcap_p.match(w2):
                            w2 = "_ALLCAP_"
                        elif lastcap_p.match(w2):
                            w2 = "_LASTCAP_"
                        else:
                            w2 = "_RARE_"

                    Qi, Qj = self.Q_map[i], self.Q_map[j] # int 2 str

                    mat[i, j, 1] = np.log(self.ngram_counts[2][('*','*',Qi)]) - np.log(self.ngram_counts[1][('*','*')])+\
                                   np.log(self.ngram_counts[2][('*',Qi,Qj)]) - np.log(self.ngram_counts[1][('*',Qi)])+\
                                   np.log(self.emission_counts.has_key((w1, Qi)) and self.emission_counts[(w1, Qi)] or 1) - np.log(self.ngram_counts[0][(Qi,)])+\
                                   np.log(self.emission_counts.has_key((w2, Qj)) and self.emission_counts[(w2, Qj)] or 1) - np.log(self.ngram_counts[0][(Qj,)])

                    mat[i, j, 1] = (self.emission_counts.has_key((w1, Qi)) and self.emission_counts.has_key((w2, Qj))) and mat[i, j, 1] or float("-inf")
            # =====================================================================
            for k in xrange(2,ncol):
                wk = sent[k][1]
                if not self.word_dict.has_key(wk):
                    if numeric_p.match(wk):
                        wk = "_NUMERIC_"
                    elif allcap_p.match(wk):
                        wk = "_ALLCAP_"
                    elif lastcap_p.match(wk):
                        wk = "_LASTCAP_"
                    else:
                        wk = "_RARE_"
                for i in xrange(0,d1):  # latent variable T - 1
                    for j in xrange(0,d2):  # latent variable T
                        Qi, Qj, Q0 = self.Q_map[i], self.Q_map[j], self.Q_map[0]  # int 2 str
                        mx = mat[0, i, k - 1] + np.log(self.ngram_counts[2][(Q0,Qi,Qj)]) - np.log(self.ngram_counts[1][(Q0,Qi)]) +\
                             np.log(self.emission_counts.has_key((wk,Qj)) and self.emission_counts[(wk, Qj)] or 1) - np.log(self.ngram_counts[0][(Qj,)])
                        mx = self.emission_counts.has_key((wk,Qj)) and mx or float("-inf")
                        mxi = 0
                        for l in xrange(1,d1):
                            Ql = self.Q_map[l]
                            pr = mat[l, i, k - 1] + np.log(self.ngram_counts[2][(Ql,Qi,Qj)]) - np.log(self.ngram_counts[1][(Ql,Qi)]) +\
                                 np.log(self.emission_counts.has_key((wk,Qj)) and self.emission_counts[(wk, Qj)] or 1) - np.log(self.ngram_counts[0][(Qj,)])
                            pr = self.emission_counts.has_key((wk,Qj)) and pr or float("-inf")
                            if pr > mx:
                                mx, mxi = pr, l
                        mat[i,j,k], matTb[i,j,k] = mx, mxi
            # =====================================================================
            """
            Calcu the final Pr which contains the "Stop" token
            """
            Pr_mat = np.zeros(shape=(d1,d2))
            for i in xrange(d1):
                for j in xrange(d2):
                    Qi, Qj = self.Q_map[i], self.Q_map[j]
                    Pr_mat[i,j] = mat[i,j,ncol-1] + np.log(self.ngram_counts[2][(Qi,Qj,'STOP')]) - np.log(self.ngram_counts[1][(Qi,Qj)])
            coordinate = np.argmax(Pr_mat)
            prb = np.max(Pr_mat)
            omxi, omxj = coordinate//d1, coordinate%d2  # int form of the Yn_1 & Yn
            # Backtrace
            p = [1] * ncol
            for k in xrange(ncol-1, 1, -1):
                omxi, omxj = matTb[omxi, omxj, k], omxi
                p.append(omxi)
            p = map(lambda x: self.Q_map[x], p[::-1])
            return p, prb

    def forward_backward_log(self, sent):
        d1, d2, ncol = len(self.Q), len(self.Q), len(sent)
        numeric_p = re.compile(r"^.*[0-9]+.*$")
        allcap_p = re.compile(r"^[A-Z]+$")
        lastcap_p = re.compile(r"^.*[A-Z]$")
        if ncol == 1:
            word = sent[0][1]
            if not self.word_dict.has_key(word):
                if numeric_p.match(word):
                    word = "_NUMERIC_"
                elif allcap_p.match(word):
                    word  = "_ALLCAP_"
                elif lastcap_p.match(word):
                    word = "_LASTCAP_"
                else:
                    word = "_RARE_"
            tagged_for_word = max(self.Q, key=lambda x: self.emission_counts[(word, x)]/self.ngram_counts[0][(x,)])
            return tagged_for_word, 0
        else:
            alpha = np.zeros(shape=(d1,d2,ncol),dtype=float)
            beta = np.zeros(shape=(d1,d2,ncol),dtype=float)
            #=====================================================================
            """
            initialize the mat matrix  
            """
            for i in xrange(0,d1):
                for j in xrange(0,d2):
                    w1, w2 = sent[0][1], sent[1][1] # 1st word & 2nd word in the sentence
                    if not self.word_dict.has_key(w1):
                        if numeric_p.match(w1):
                            w1 = "_NUMERIC_"
                        elif allcap_p.match(w1):
                            w1 = "_ALLCAP_"
                        elif lastcap_p.match(w1):
                            w1 = "_LASTCAP_"
                        else:
                            w1 = "_RARE_"
                    if not self.word_dict.has_key(w2):
                        if numeric_p.match(w2):
                            w2 = "_NUMERIC_"
                        elif allcap_p.match(w2):
                            w2 = "_ALLCAP_"
                        elif lastcap_p.match(w2):
                            w2 = "_LASTCAP_"
                        else:
                            w2 = "_RARE_"

                    Qi, Qj = self.Q_map[i], self.Q_map[j] # int 2 str

                    alpha[i, j, 1] = np.log(self.ngram_counts[2][('*','*',Qi)]) - np.log(self.ngram_counts[1][('*','*')])+\
                                   np.log(self.ngram_counts[2][('*',Qi,Qj)]) - np.log(self.ngram_counts[1][('*',Qi)])+\
                                   np.log(self.emission_counts.has_key((w1, Qi)) and self.emission_counts[(w1, Qi)] or 1) - np.log(self.ngram_counts[0][(Qi,)])+\
                                   np.log(self.emission_counts.has_key((w2, Qj)) and self.emission_counts[(w2, Qj)] or 1) - np.log(self.ngram_counts[0][(Qj,)])

                    alpha[i, j, 1] = (self.emission_counts.has_key((w1, Qi)) and self.emission_counts.has_key((w2, Qj))) and alpha[i, j, 1] or float("-inf")
            for i in xrange(d1):
                for j in xrange(d2):
                    Qi, Qj = self.Q_map[i], self.Q_map[j]
                    beta[i,j,ncol-1] = np.log(self.ngram_counts[2][(Qi,Qj,'STOP')]) - np.log(self.ngram_counts[1][(Qi,Qj)])
            # =====================================================================
            """
            Calcu alpha_ij(t) and beta_ij(t)
            """
            for k in xrange(2,ncol):
                wk = sent[k][1]
                if not self.word_dict.has_key(wk):
                    if numeric_p.match(wk):
                        wk = "_NUMERIC_"
                    elif allcap_p.match(wk):
                        wk = "_ALLCAP_"
                    elif lastcap_p.match(wk):
                        wk = "_LASTCAP_"
                    else:
                        wk = "_RARE_"
                for i in xrange(d1):
                    for j in xrange(d2):
                        Qi, Qj, Q0 = self.Q_map[i], self.Q_map[j], self.Q_map[0]
                        mid = alpha[0,i,k-1] + np.log(self.ngram_counts[2][(Q0,Qi,Qj)]) - np.log(self.ngram_counts[1][(Q0,Qi)])
                        for l in xrange(1,d1):
                            Ql = self.Q_map[l]
                            power = alpha[l,i,k-1] + np.log(self.ngram_counts[2][(Ql,Qi,Qj)]) - np.log(self.ngram_counts[1][(Ql,Qi)])
                            mid = logadd(mid, power)
                        alpha[i,j,k] = mid + np.log(self.emission_counts.has_key((wk,Qj)) and self.emission_counts[(wk, Qj)] or 1)\
                                       - np.log(self.ngram_counts[0][(Qj,)])
                        alpha[i, j, k] = self.emission_counts.has_key((wk, Qj)) and alpha[i, j, k] or float("-inf")
            for k in xrange(ncol-2, 0, -1):
                wk_puls_1 = sent[k+1][1]
                if not self.word_dict.has_key(wk_puls_1):
                    if numeric_p.match(wk_puls_1):
                        wk_puls_1 = "_NUMERIC_"
                    elif allcap_p.match(wk_puls_1):
                        wk_puls_1 = "_ALLCAP_"
                    elif lastcap_p.match(wk_puls_1):
                        wk_puls_1 = "_LASTCAP_"
                    else:
                        wk_puls_1 = "_RARE_"
                for i in xrange(d1):
                    for j in xrange(d2):
                        Qi, Qj, Q0 = self.Q_map[i], self.Q_map[j], self.Q_map[0]
                        mid = beta[j,0,k+1] + np.log(self.ngram_counts[2][(Qi,Qj,Q0)]) - np.log(self.ngram_counts[1][(Qi,Qj)])+\
                            np.log(self.emission_counts.has_key((wk_puls_1,Q0)) and self.emission_counts[(wk_puls_1,Q0)] or 1) - np.log(self.ngram_counts[0][(Qj,)])
                        mid = self.emission_counts.has_key((wk_puls_1,Q0)) and mid or float("-inf")
                        for l in xrange(1,d1):
                            Ql = self.Q_map[l]
                            power = beta[j, l, k + 1] + np.log(self.ngram_counts[2][(Qi, Qj, Ql)]) - np.log(
                                self.ngram_counts[1][(Qi, Qj)]) + \
                                    np.log(self.emission_counts.has_key((wk_puls_1, Ql)) and self.emission_counts[
                                        (wk_puls_1, Ql)] or 1) - np.log(self.ngram_counts[0][(Qj,)])
                            power = self.emission_counts.has_key((wk_puls_1, Ql)) and power or float("-inf")
                            mid = logadd(mid, power)
                        beta[i,j,k] = mid
            # =====================================================================
            # =====================================================================
            """
            Calcu the final Pr which contains the "Stop" token
            """
            Pr = np.zeros([d1,1])
            for i in xrange(d1):
                Qi, Q0 = self.Q_map[i], self.Q_map[0]
                mid = alpha[0,i,ncol-1] + np.log(self.ngram_counts[2][(Q0,Qi,'STOP')]) - np.log(self.ngram_counts[1][(Q0,Qi)])
                for l in xrange(1,d1):
                    Ql = self.Q_map[l]
                    power = alpha[l,i,ncol-1] + np.log(self.ngram_counts[2][(Ql,Qi,'STOP')]) - np.log(self.ngram_counts[1][(Ql,Qi)])
                    mid = logadd(mid,power)
                Pr[i,0] = mid
            # =====================================================================
            marginPr = np.zeros_like(alpha)
            p, prb = [], []
            for t in xrange(1,ncol):
                marginPr[:,:,t] = alpha[:,:,t] + beta[:,:,t]
                max_val, mxi = float("-inf"), 0
                for i in xrange(d1):
                    cur_val = float("-inf")
                    for j in xrange(d2):
                        cur_val = logadd(cur_val, marginPr[i,j,t])
                    if cur_val > max_val:
                        max_val = cur_val
                        mxi = i
                p.append(mxi)

            index = np.argmax(Pr)
            p.append(index)

            p = map(lambda x: self.Q_map[x], p)

            return p







if __name__ == "__main__":
    f_counts = file('./fixed_gene.counts', 'r')
    f_test = file('./gene.test', 'r')
    #f_tagged = file('./gene_test.p2.out', 'w')   # file tagged by viterbi algo
    f_tagged = file('./gene_test.p4.out', 'w') # file tagged by forward_backward algo
    hmm = Hmm()
    hmm.read_counts(f_counts)
    f_counts.close()


    sent_iterator = sentence_iterator(simple_conll_corpus_iterator(f_test))
    for sent in sent_iterator:
        #tag_list, _ = hmm.ViterbiLog(sent)
        tag_list = hmm.forward_backward_log(sent)

        for word, tag in zip(sent, tag_list):
            word = word[1]
            _l = word + " " + tag + "\n"
            f_tagged.write(_l)
        f_tagged.write("\n")