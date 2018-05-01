import string
import math
import re

# compute log(1+x) without loss of precision for small values of x
def log_one_plus_x(x):
    if x <= -1.0:
        raise FloatingPointError, "argument must be > -1"
    if abs(x) > 1e-4:
        # x is large enough that the obvious evaluation is OK
        return math.log(1.0 + x)
    else:
        # Use Taylor approx. 
        # log(1 + x) = x - x^2/2 with error roughly x^3/3
        # Since |x| < 10^-4, |x|^3 < 10^-12, 
        # and the relative error is less than 10^-8
        return (-0.5*x + 1.0)*x
def log_add(x, y):
    if y < x:
        return x + log_one_plus_x(math.exp(y-x))
    else:
        return y + log_one_plus_x(math.exp(x-y))
class emission_parameters:
    def __init__(self, file_path):
        print "initializing..."
        self.xy_dict = {}
        self.gram1_dict = {}
        self.gram2_dict = {}
        self.gram3_dict = {}
        self.words_dict = {}
        self.labels = []
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            words = line.split()
            if len(words) < 3:
                continue
            if words[1] == 'WORDTAG':
                if not self.xy_dict.has_key(words[2]):
                    self.xy_dict[words[2]] = {}
                if not self.xy_dict[words[2]].has_key(words[3]):
                    self.xy_dict[words[2]][words[3]] = 0
                self.xy_dict[words[2]][words[3]] += string.atoi(words[0])
                if words[2] not in self.labels:
                    self.labels.append(words[2])
                if words[3] not in self.words_dict:
                    self.words_dict[words[3]] = 0
            elif words[1] == '1-GRAM':
                if not self.gram1_dict.has_key(words[2]):
                    self.gram1_dict[words[2]] = string.atoi(words[0])
                else:
                    self.gram1_dict[words[2]] += string.atoi(words[0])
            elif words[1] == '2-GRAM':
                if not self.gram2_dict.has_key(words[2]):
                    self.gram2_dict[words[2]] = {}
                if not self.gram2_dict[words[2]].has_key(words[3]):
                    self.gram2_dict[words[2]][words[3]] = string.atoi(words[0])
                else:
                    self.gram2_dict[words[2]][words[3]] += string.atoi(words[0])
            elif words[1] == '3-GRAM':
                if not self.gram3_dict.has_key(words[2]):
                    self.gram3_dict[words[2]] = {}
                if not self.gram3_dict[words[2]].has_key(words[3]):
                    self.gram3_dict[words[2]][words[3]] = {}
                if not self.gram3_dict[words[2]][words[3]].has_key(words[4]):
                    self.gram3_dict[words[2]][words[3]][words[4]] = string.atoi(words[0])
                else:
                    self.gram3_dict[words[2]][words[3]][words[4]] += string.atoi(words[0])
        # print "finish geting xy_dict, %d entries" % len(self.xy_dict)
    def test(self, test_path, test_out_path):
        f = open(test_path, 'r')
        lines = f.readlines()
        f.close()
        lines2 = []
        for line in lines:
            line2 = line
            words = line.split()
            if len(words) > 0:
                flag = False
                x = words[0]
                for y in self.xy_dict:
                    if self.xy_dict[y].has_key(x):
                        if flag:
                            if float(self.xy_dict[y][x]) / self.gram1_dict[y] > max_prob:
                                max_prob = float(self.xy_dict[y][x]) / self.gram1_dict[y]
                                max_label = y
                        else:
                            max_prob = float(self.xy_dict[y][x]) / self.gram1_dict[y]
                            max_label = y
                            flag = True
                if not flag:
                    x = "_RARE_"
                    for y in self.xy_dict:
                        if self.xy_dict[y].has_key(x):
                            if flag:
                                if float(self.xy_dict[y][x]) / self.gram1_dict[y] > max_prob:
                                    max_prob = float(self.xy_dict[y][x]) / self.gram1_dict[y]
                                    max_label = y
                            else:
                                max_prob = float(self.xy_dict[y][x]) / self.gram1_dict[y]
                                max_label = y
                                flag = True
                assert flag == True
                line2 = words[0] + " " + max_label + "\n"
            lines2.append(line2)
        f = open(test_out_path, "w")
        f.writelines(lines2)
        f.close()
    def classify(self, classify_path, classify_out_path):
        f = open(classify_path, 'r')
        lines = f.readlines()
        f.close()
        lines2 = []
        sentence = []
        # print 'start classifying using hmm..'
        for line in lines:
            line2 = line
            if len(line) <= 1:
                max_labels = self.classify_sentence(sentence)
                for i in range(0,len(sentence)):
                    words = sentence[i].split()
                    line2 = words[0] + " " + max_labels[i] + "\n"
                    lines2.append(line2)
                lines2.append("\n")
                #print sentence
                #print max_labels
                sentence = []
            else:
                sentence.append(line)
        f = open(classify_out_path, 'w')
        f.writelines(lines2)
        f.close()
            
        # print 'classification finished'
    def classify_sentence(self, sentence):
        #print sentence
        assert len(sentence) >= 2
        numeric_p = re.compile(r"^.*[0-9]+.*$")
        allcap_p = re.compile(r"^[A-Z]+$")
        lastcap_p = re.compile(r"^.*[A-Z]$")
        viterbi_state = []
        viterbi_labels = []
        viterbi_state.append({})
        viterbi_labels.append({})
        #for label in self.labels:
        #    viterbi_state[0][label] = float(self.gram3_dict['*']['*'][label]) / self.gram2_dict['*']['*'] * self.xy_dict[label][sentence[0].split()[0]]
        #    viterbi_labels.append(max(viterbi_state[0].items(), key=lamda x:x[1])[0])
        viterbi_state[0]['*'] = {}
        viterbi_labels[0]['*'] = {}
        for label in self.labels:
            viterbi_labels[0]['*'][label] = '*'
            word = sentence[0].split()[0]
            if not self.words_dict.has_key(word):
                if numeric_p.match(word):
                    word = '_NUMERIC_'
                elif allcap_p.match(word):
                    word = '_ALLCAP_'
                elif lastcap_p.match(word):
                    word = '_LASTCAP_'
                else:
                    word = '_RARE_'
            viterbi_state[0]['*'][label] = math.log(self.gram3_dict['*']['*'][label]) - math.log(self.gram2_dict['*']['*']) + math.log((self.xy_dict[label].has_key(word) and self.xy_dict[label][word] or 1)) - math.log(self.gram1_dict[label])
            viterbi_state[0]['*'][label] = self.xy_dict[label].has_key(word) and viterbi_state[0]['*'][label] or float('-inf')
        viterbi_state.append({})
        viterbi_labels.append({})
        for label in self.labels:
            if not viterbi_state[1].has_key(label):
                viterbi_state[1][label] = {}
            if not viterbi_labels[1].has_key(label):
                viterbi_labels[1][label] = {}
            for label2 in self.labels:
                word = sentence[1].split()[0]
                if not self.words_dict.has_key(word):
                    if numeric_p.match(word):
                        word = '_NUMERIC_'
                    elif allcap_p.match(word):
                        word = '_ALLCAP_'
                    elif lastcap_p.match(word):
                        word = '_LASTCAP_'
                    else:
                        word = '_RARE_'
                viterbi_state[1][label][label2] = viterbi_state[0]['*'][label] + math.log(self.gram3_dict['*'][label][label2]) - math.log(self.gram2_dict['*'][label]) + math.log(self.xy_dict[label2].has_key(word) and self.xy_dict[label2][word] or 1) - math.log(self.gram1_dict[label2])
                viterbi_state[1][label][label2] = self.xy_dict[label2].has_key(word) and viterbi_state[1][label][label2] or float('-inf')
                viterbi_labels[1][label][label2] = '*'
                #print 1,label,label2
                #print viterbi_state[1][label][label2]
        for i in range(2,len(sentence)):
            viterbi_state.append({})
            viterbi_labels.append({})
            for label in self.labels:
                if not viterbi_state[i].has_key(label):
                    viterbi_state[i][label] = {}
                if not viterbi_labels[i].has_key(label):
                    viterbi_labels[i][label] = {}
                for label2 in self.labels:
                    flag = False
                    word = sentence[i].split()[0]
                    if not self.words_dict.has_key(word):
                        if numeric_p.match(word):
                            word = '_NUMERIC_'
                        elif allcap_p.match(word):
                            word = '_ALLCAP_'
                        elif lastcap_p.match(word):
                            word = '_LASTCAP_'
                        else:
                            word = '_RARE_'
                    for label0 in self.labels:
                        if flag:
                            viterbi_p = viterbi_state[i-1][label0][label] + math.log(self.gram3_dict[label0][label][label2]) - math.log(self.gram2_dict[label0][label]) + math.log(self.xy_dict[label2].has_key(word) and self.xy_dict[label2][word] or 1) - math.log(self.gram1_dict[label2])
                            viterbi_p = self.xy_dict[label2].has_key(word) and viterbi_p or float('-inf')
                            if viterbi_p > max_viterbi_p:
                                max_viterbi_p = viterbi_p
                                max_label0 = label0
                        else:
                            flag = True
                            viterbi_p = viterbi_state[i-1][label0][label] + math.log(self.gram3_dict[label0][label][label2]) - math.log(self.gram2_dict[label0][label]) + math.log(self.xy_dict[label2].has_key(word) and self.xy_dict[label2][word] or 1) - math.log(self.gram1_dict[label2])
                            viterbi_p = self.xy_dict[label2].has_key(word) and viterbi_p or float('-inf')
                            max_viterbi_p = viterbi_p
                            max_label0 = label0
                    viterbi_state[i][label][label2] = max_viterbi_p
                    viterbi_labels[i][label][label2] = max_label0
                    # print i,label,label2
                    # print max_viterbi_p

        i = len(sentence)
        viterbi_state.append({})
        viterbi_labels.append({})
        label2 = 'STOP'
        for label in self.labels:
            if not viterbi_state[i].has_key(label):
                viterbi_state[i][label] = {}
            if not viterbi_labels[i].has_key(label):
                viterbi_labels[i][label] = {}
            flag = False
            for label0 in self.labels:
                if flag:
                    viterbi_p = viterbi_state[i-1][label0][label] + math.log(self.gram3_dict[label0][label][label2]) - math.log(self.gram2_dict[label0][label])
                    if viterbi_p > max_viterbi_p:
                        max_viterbi_p = viterbi_p
                        max_label0 = label0
                else:
                    flag = True
                    viterbi_p = viterbi_state[i-1][label0][label] + math.log(self.gram3_dict[label0][label][label2]) - math.log(self.gram2_dict[label0][label])
                    max_viterbi_p = viterbi_p
                    max_label0 = label0
            viterbi_state[i][label][label2] = max_viterbi_p
            viterbi_labels[i][label][label2] = max_label0

        max_labels = [0]*len(sentence)
        flag = False
        for label in self.labels:
            if flag:
                if viterbi_state[i][label][label2] > max_viterbi_p:
                    max_viterbi_p = viterbi_state[i][label][label2]
                    max_label = label
            else:
                flag = True
                max_viterbi_p = viterbi_state[i][label][label2]
                max_label = label
        label = max_label
        label0 = viterbi_labels[i][label][label2]
        for i in range(len(sentence)-1, -1, -1):
            max_labels[i] = label
            label2 = label
            label = label0
            label0 = viterbi_labels[i][label][label2]
        return max_labels

    def decode_posterior(self, classify_path, classify_out_path):
        f = open(classify_path, 'r')
        lines = f.readlines()
        f.close()
        lines2 = []
        sentence = []
        # print 'start posterior decoding using hmm..'
        j=0
        for line in lines:
            line2 = line
            if len(line) <= 1:
                max_labels = self.decode_posterior_sentence(sentence)
                j = j+1
                #if j > 5:
                #    break
                for i in range(0,len(sentence)):
                    words = sentence[i].split()
                    line2 = words[0] + " " + max_labels[i] + "\n"
                    lines2.append(line2)
                lines2.append("\n")
                #print sentence
                #print max_labels
                sentence = []
            else:
                sentence.append(line.strip())
        f = open(classify_out_path, 'w')
        f.writelines(lines2)
        f.close()
            
        # print 'classification finished'
    def decode_posterior_sentence(self, sentence):
        #print sentence
        n = len(sentence)
        assert len(sentence) >= 2
        numeric_p = re.compile(r"^.*[0-9]+.*$")
        allcap_p = re.compile(r"^[A-Z]+$")
        lastcap_p = re.compile(r"^.*[A-Z]$")

        sentence_full = {}
        sentence_full[-1] = '*'
        sentence_full[0] = '*'
        for i in range(len(sentence)):
            word = sentence[i]
            if not self.words_dict.has_key(word):
                if numeric_p.match(word):
                    word = '_NUMERIC_'
                elif allcap_p.match(word):
                    word = '_ALLCAP_'
                elif lastcap_p.match(word):
                    word = '_LASTCAP_'
                else:
                    word = '_RARE_'
            sentence_full[i+1] = word

        sentence_full[n+1] = '_NOT_EXIST_'
        alpha = {}
        for label1 in self.labels:
            for label2 in self.labels:
                alpha[(0, label1, label2)] = math.log(self.gram3_dict['*']['*'][label1]) + math.log(self.gram3_dict['*'][label1][label2]) - math.log(self.gram2_dict['*'][label1])
        for i in range(1, n):
            word = sentence_full[i]
            for label1 in self.labels:
                for label2 in self.labels:
                    for label in self.labels:
                        if (i, label1, label2) not in alpha:
                            alpha[(i, label1, label2)] = alpha[(i-1, label, label1)]+math.log(self.gram3_dict[label][label1][label2])-math.log(self.gram2_dict[label][label1]) + math.log(self.xy_dict[label].has_key(word) and self.xy_dict[label][word] or 1) - math.log(self.gram1_dict[label])
                        else:
                            alpha[(i, label1, label2)] = log_add(alpha[(i, label1, label2)], alpha[(i-1,label,label1)]+math.log(self.gram3_dict[label][label1][label2])-math.log(self.gram2_dict[label][label1]) + math.log(self.xy_dict[label].has_key(word) and self.xy_dict[label][word] or 1) - math.log(self.gram1_dict[label]))
        #print alpha[(0,'O','O')]
        #print alpha[(0,'I-GENE','I-GENE')]
        #print alpha[(1,'O','O')]
        #print alpha[(1,'I-GENE','I-GENE')]
        #print alpha[(2,'O','O')]
        #print alpha[(2,'I-GENE','I-GENE')]
        #print alpha[(3,'O','O')]
        #print alpha[(3,'I-GENE','I-GENE')]
        beta = {}
        for label1 in self.labels:
            for label2 in self.labels:
                beta[(n+2, label1, label2)] = 0
                beta[(n+1, label1, label2)] = math.log(self.gram3_dict[label1][label2]['STOP']) - math.log(self.gram2_dict[label1][label2])
        for i in range(n, 2, -1):
            word = sentence_full[i]
            for label1 in self.labels:
                for label2 in self.labels:
                    for label in self.labels:
                        if (i, label1, label2) not in beta:
                            beta[(i, label1, label2)] = beta[(i+1, label2, label)]+math.log(self.gram3_dict[label1][label2][label])-math.log(self.gram2_dict[label1][label2]) + math.log(self.xy_dict[label].has_key(word) and self.xy_dict[label][word] or 1) - math.log(self.gram1_dict[label])
                        else:
                            beta[(i, label1, label2)] = log_add(beta[(i,label1,label2)],beta[(i+1, label2, label)]+math.log(self.gram3_dict[label1][label2][label])-math.log(self.gram2_dict[label1][label2]) + math.log(self.xy_dict[label].has_key(word) and self.xy_dict[label][word] or 1) - math.log(self.gram1_dict[label]))
        max_labels = []
        for i in range(1,n+1):
            word = sentence_full[i]
            max_val = -float('inf')
            max_label = ''
            for label in self.labels:
                curr_val = -float('inf')
                for label2 in self.labels:
                    curr_val = log_add(curr_val, alpha[(i-1,label,label2)]+beta[(i+2,label,label2)]+math.log(self.xy_dict[label].has_key(word) and self.xy_dict[label][word] or 1) - math.log(self.gram1_dict[label])+math.log(self.xy_dict[label2].has_key(sentence_full[i+1]) and self.xy_dict[label2][sentence_full[i+1]] or 1) - math.log(self.gram1_dict[label2]))
                #print label
                #print curr_val
                if curr_val > max_val:
                    max_label = label
                    max_val = curr_val
            max_labels.append(max_label)
        #print max_labels
        return max_labels

if __name__ == "__main__":
    counts_path = "./fixed_gene.counts"
    test_path = "./gene.test"
    test_out_path = './gene_test.p2.out'
    epara = emission_parameters(counts_path)
    epara.classify(test_path, test_out_path)
