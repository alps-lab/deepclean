#!/usr/bin/env python
import csv
from collections import defaultdict

import ujson
from plyvel import DB

CATEGORYLINKS_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-categorylinks-clean.csv"
PREFIX = b"category_pageids_"


if __name__ == "__main__":
    db = None

    cate_pageids = defaultdict(list)
    with open(CATEGORYLINKS_PATH) as fobj:
        reader = csv.reader(fobj)
        for row in reader:
            if row[2] != 'page':
                continue

            page_id, cat_name = row[0], row[1]
            cate_pageids[row[1]].append(row[0])

    try:
        db = DB("/home/xinyang/Datasets/Wikipedia/enwiki_leveldb")
        for cate_name, page_ids in cate_pageids.items():
            db.put(PREFIX + cate_name.encode(), ujson.dumps(page_ids).encode())
    finally:
        db.close()