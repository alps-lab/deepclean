#!/usr/bin/env python
import csv
import numpy as np


IN_PATH = "/home/xinyang/Datasets/dataclean/tables/DBPedia/prototype/new_Economist.csv"


def pick_random(ls, num):
    n = len(ls)
    indices = np.random.choice(np.arange(n), num, replace=False)

    return [ls[i] for i in indices]


def denormalize(s):
    if "(" in s:
        return s.split("(")[0].strip()
    else:
        return s


def main():
    with open(IN_PATH, encoding="latin-1") as fobj:
        reader = csv.reader(fobj)
        header = next(reader)

        rows = [row for row in reader]

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/200", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1]

        to_output = []
        for row in rows:
            if row[3] == "United States":
                to_output.append(row)

        to_output = pick_random(to_output, 42)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(denormalize(row[i]) for i in use_cols)

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/201", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 3]

        to_output = []
        for row in rows:
            if 1930 <= int(row[2]):
                to_output.append(row)

        to_output = pick_random(to_output, 45)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(row[i] for i in use_cols)


    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/202", "w") as fobj:
        writer = csv.writer(fobj)

        to_output = pick_random(rows, 55)

        writer.writerow(header)
        for row in to_output:
            writer.writerow(row)


if __name__ == "__main__":
    main()
