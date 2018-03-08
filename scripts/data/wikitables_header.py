#!/usr/bin/env python
import os

from collections import OrderedDict

DATA_DIR = "/home/xinyang/Datasets/dataclean/EX/tablesForAnnotation/wikitables/"

CANDIDATES = OrderedDict()


if __name__ == "__main__":
    name_headers = OrderedDict()
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".txt"):
            continue
        with open(os.path.join(DATA_DIR, name)) as fobj:
            it = iter(fobj)
            name_headers[name] = next(it).strip()

    for name, header in name_headers.items():
        print("%s:    %s" % (name, header))

    candidates = OrderedDict()

    def add(name):
        candidates[name] = name_headers[name]

    def print_all():
        for name, header in candidates.items():
            print("%s:    %s" % (name, header))