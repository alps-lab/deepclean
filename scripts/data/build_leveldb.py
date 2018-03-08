#!/usr/bin/env python
import sqlite3
import json
import csv
from collections import defaultdict

from tqdm import tqdm
import plyvel


from theclean.utils.wiki_utils import category_filter

PREFIX_ID_TITLE = b"id_title_"
PREFIX_TITLE_ID = b"title_id_"
PREFIX_ID_DOC = b"id_doc_"
PREFIX_ID_CATEGORIES = b"id_category_"
PREFIX_CATEGORY_IDS = b"category_id_"


if __name__ == "__main__":
    sqlite_db, level_db = None, None
    try:
        sqlite_db = sqlite3.connect("/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa.db")
        level_db = plyvel.DB("/home/xinyang/Datasets/Wikipedia/enwiki_leveldb/", create_if_missing=True)

        cursor = sqlite_db.execute("SELECT * FROM documents")

        good_ids = set()
        for row in tqdm(cursor):
            good_ids.add(row[2])
            level_db.put(PREFIX_ID_TITLE + row[2].encode(), row[0].encode())
            level_db.put(PREFIX_TITLE_ID + row[0].encode(), row[2].encode())
            level_db.put(PREFIX_ID_DOC + row[2].encode(), row[1].encode())

        id_categories = {}
        for good_id in good_ids:
            id_categories[good_id] = []

        category_ids = defaultdict(list)

        with open("/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-categorylinks-clean.csv") as fobj:
            reader = csv.reader(fobj)
            for doc_id, cate_name, _ in tqdm(reader):
                if doc_id not in good_ids:
                    continue
                if not category_filter(cate_name):
                    continue
                category_ids[cate_name].append(doc_id)
                id_categories[doc_id].append(cate_name)

        for cate_name, doc_ids in tqdm(category_ids.items()):
            level_db.put(PREFIX_CATEGORY_IDS + cate_name.encode(), json.dumps(doc_ids).encode())

        for doc_id, cats in tqdm(id_categories.items()):
            level_db.put(PREFIX_ID_CATEGORIES + doc_id.encode(),
                         json.dumps(cats).encode())

    finally:
        if sqlite_db:
            sqlite_db.close()
        if level_db:
            level_db.close()

