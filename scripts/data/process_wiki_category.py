#!/usr/bin/env python
import sqlite3
import csv


CATEGORY_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-category.csv"
CATEGORYLINKS_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-categorylinks.csv"
OUTPUT_PATH = "/home/xinyang/Datasets/Wikipedia/wikipedia_20180101_categoryinfo.db"


SQL1 = """
    CREATE TABLE category(
      cat_id INTEGER(10) PRIMARY KEY,
      cat_title VARCHAR(255)
    );
"""

SQL2 = """
    CREATE TABLE categorylinks(
      cl_from INTEGER(10),
      cl_to VARCHAR(255)
    );
"""

SQL3 = """
    INSERT INTO category (cat_id, cat_title)
    VALUES (?, ?);
"""


SQL4 = """
    INSERT INTO categorylinks (cl_from, cl_to) 
    VALUES (?, ?);
"""


def main():
    conn = sqlite3.connect(OUTPUT_PATH)

    cursor = conn.execute(SQL1)
    cursor.close()

    cursor = conn.execute(SQL2)
    cursor.close()

    with open(CATEGORY_PATH, encoding="latin1") as fobj:
        reader = csv.reader(fobj)
        rows = []

        for i, row in enumerate(reader):
            cat_id, cat_title = row[:2]
            cat_id = int(cat_id)
            rows.append((cat_id, cat_title))

        cursor = conn.executemany(SQL3, rows)
        conn.commit()
        cursor.close()

    with open(CATEGORYLINKS_PATH, encoding="latin1") as fobj:
        reader = csv.reader(fobj)
        rows = []
        for row in reader:
            cl_from, cl_to = row[:2]
            cl_from = int(cl_from)
            rows.append((cl_from, cl_to))

        cursor = conn.executemany(SQL4, rows)
        conn.commit()
        cursor.close()


if __name__ == "__main__":
    main()