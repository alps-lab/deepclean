#!/usr/bin/env python
import csv
import numpy as np


IN_PATH = "/home/xinyang/Datasets/dataclean/tables/DBPedia/prototype/new_Nobel.csv"


def pick_random(ls, num):
    n = len(ls)
    indices = np.random.choice(np.arange(n), num, replace=False)

    return [ls[i] for i in indices]


def main():
    with open(IN_PATH, encoding="latin-1") as fobj:
        reader = csv.reader(fobj)
        header = next(reader)

        rows = [row for row in reader]

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/100", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 3]

        to_output = []
        for row in rows:
            if 1995 <= int(row[0]) <= 2005:
                to_output.append(row)

        to_output = pick_random(to_output, 65)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(row[i] for i in use_cols)

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/101", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 3]

        prizes = ["The Nobel Prize in Physiology or Medicine", "The Nobel Prize in Chemistry",
                  "The Nobel Prize in Physics"]
        to_output = []
        for row in rows:
            if 1940 <= int(row[0]) <= 1990 and row[3] in prizes:
                to_output.append(row)

        to_output = pick_random(to_output, 70)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(row[i] for i in use_cols)


    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/102", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 3]

        prizes = ["The Nobel Prize in Literature", "The Nobel Peace Prize"]
        to_output = []
        for row in rows:
            if 1970 <= int(row[0]) <= 2000 and row[3] in prizes:
                to_output.append(row)

        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(row[i] for i in use_cols)

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/103", "w") as fobj:
        writer = csv.writer(fobj)

        to_output = pick_random(rows, 90)

        writer.writerow(header)
        for row in to_output:
            writer.writerow(row)


if __name__ == "__main__":
    main()
