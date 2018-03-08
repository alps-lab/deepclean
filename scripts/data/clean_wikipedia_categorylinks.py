#!/usr/bin/env
import csv


INPUT_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-categorylinks.csv"
OUTPUT_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-categorylinks-clean.csv"


if __name__ == "__main__":
    with open(INPUT_PATH, encoding="latin1") as fin, open(OUTPUT_PATH, "w") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            row = [row[0], row[1], row[-1]]
            writer.writerow(row)
