#!/usr/bin/env python
import os, sys,re
from emission_parameters import emission_parameters

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: python %s <counts-file> <test-file> <output-file>" %sys.argv[0]
        print >> sys.stderr, "Posterior decoding"
        sys.exit(1)
    counts_path = sys.argv[1]
    test_path = sys.argv[2]
    test_out_path = sys.argv[3]
    epara = emission_parameters(counts_path)
    epara.decode_posterior(test_path, test_out_path)
