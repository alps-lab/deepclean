#!/usr/bin/env python
import csv

from tqdm import tqdm


DATA_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-page.csv"
OUTPUT_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-page-clean.csv"


if __name__ == "__main__":
    with open(DATA_PATH, encoding="latin1") as fin, open(OUTPUT_PATH, "w") as fout:
        reader = csv.reader((line.replace("\0", "") for line in fin))
        writer = csv.writer(fout)

        for row in tqdm(reader):
            row = row[:3]
            writer.writerow(row)
