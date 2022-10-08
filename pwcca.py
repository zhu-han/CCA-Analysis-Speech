#!/usr/bin/env python3 -u

import sys
import numpy as np
import ecco.analysis as analysis

def main():
    np1 = np.load(sys.argv[1])
    np2 = np.load(sys.argv[2])
    np1 = np.transpose(np1[:195213], (1, 0))
    np2 = np.transpose(np2[:195213], (1, 0))
    similarity = 1/2 * (analysis.pwcca(np1, np2) + analysis.pwcca(np2, np1))
    print(similarity)

if __name__ == "__main__":
    main()

