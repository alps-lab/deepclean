#!/usr/bin/env python
import csv
import ujson

from tqdm import tqdm
from plyvel import DB


DATA_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-category.csv"
PREFIX = b"category_counts_"


def main():
    db = DB("/home/xinyang/Datasets/Wikipedia/enwiki_leveldb/")
    reader = csv.reader(open(DATA_PATH, encoding="latin1"))

    for pid, name, num_pages, num_subcats, num_files in tqdm(reader):
        name = name.encode()
        dobj = dict(num_pages=int(num_pages), num_subcats=int(num_subcats),
                    num_files=int(num_files))
        dobj_dumped = ujson.dumps(dobj).encode()

        db.put(PREFIX + name, dobj_dumped)


if __name__ == "__main__":
    main()
