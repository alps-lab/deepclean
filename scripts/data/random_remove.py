#!/usr/bin/env python
from six.moves import xrange

import csv
import argparse

import numpy as np


EMPTY_TAG = "<E>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_doc")
    parser.add_argument("out_doc")
    parser.add_argument("-f", "--fraction", dest="fraction", type=float,
                        default=0.3)
    parser.add_argument("-s", "--seed", dest="seed", type=int)

    config = parser.parse_args()
    rnd = np.random.RandomState(config.seed)

    items = []
    with open(config.in_doc) as fobj:
        reader = csv.reader(fobj)
        for item in reader:
            items.append(item)

    header, items = items[0], items[1:]
    with open(config.out_doc, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(header)

        for item in items:
            for i in xrange(len(item)):
                if rnd.uniform() < config.fraction:
                    item[i] = EMPTY_TAG
            writer.writerow(item)

