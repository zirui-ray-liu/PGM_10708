#!/usr/bin/env python
import os, sys

def countWords(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    words_dict = {}
    for line in lines:
        words = line.split()
        if len(words) == 2:
            if words_dict.has_key(words[0]):
                words_dict[words[0]] += 1
            else:
                words_dict[words[0]] = 1
    return words_dict

def replaceInfreqWords(file_path, out_path, words_dict, freq_min):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    lines2 = []
    for line in lines:
        line2 = line
        words = line.split()
        if len(words) == 2:
            if words_dict[words[0]] < freq_min:
                line2 = "_RARE_ " + words[1] + "\n"
        lines2.append(line2)
    f = open(out_path, "w")
    f.writelines(lines2)
    f.close()
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: python %s <original-file> <output-file>" %sys.argv[0]
        print >> sys.stderr, "Replace rare words (counts < 5) with _RARE_"
        sys.exit(1)
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    words_dict = countWords(file_path)
    replaceInfreqWords(file_path, out_path, words_dict, 5)
