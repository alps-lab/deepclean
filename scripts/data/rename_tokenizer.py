#!/usr/bin/env python
import numpy as np

INPUT_PATH = "/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa-tfidf-ngram=2-hash=16777216-tokenizer=corenlp.npz"
OUTPUT_PATH = "/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa-tfidf-ngram=2-hash=16777216-tokenizer=corenlpserver.npz"



if __name__ == "__main__":
    obj = np.load(INPUT_PATH)
    d = {}
    for key, value in obj.items():
        d[key] = value

    d["metadata"].item(0)["tokenizer"] = "corenlpserver"

    np.savez(OUTPUT_PATH, **d)