#!/usr/bin/env python
import csv
import numpy as np


IN_PATH = "/home/xinyang/Datasets/dataclean/tables/DBPedia/prototype/new_Currency.csv"


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

        rows = [row for row in reader if row[1] != "NULL"]
        single_rows = [row for row in rows if not row[1].startswith("{")]
        multiple_rows = [row for row in rows if row[1].startswith("{")]

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/300", "w") as fobj:
        writer = csv.writer(fobj)

        to_output = single_rows

        to_output = pick_random(to_output, 45)
        writer.writerow(header)
        for row in to_output:
            writer.writerow(row)

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/301", "w") as fobj:
        writer = csv.writer(fobj)
        to_output = []
        for row in single_rows:
            if "(" in row[0]:
                to_output.append([denormalize(row[0]), row[1]])
            else:
                to_output.append(row)

        to_output = pick_random(to_output, 45)
        writer.writerow(header)
        for row in to_output:
            writer.writerow(row)


    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/302", "w") as fobj:
        writer = csv.writer(fobj)

        to_output = []
        fractions = np.random.uniform(0.08, 0.16)
        num_multiple = int(60 * fractions)

        for row in multiple_rows:
            countries = row[1][1:-1].split("|")
            countries = [denormalize(name) for name in countries]
            to_output.append([denormalize(row[0]), "; ".join(countries)])

        # print(multiple_rows, to_output)
        to_output = pick_random(to_output, min(num_multiple, len(to_output)))

        singles = pick_random(single_rows, 60 - len(to_output))
        for row in singles:
            to_output.append([denormalize(row[0]), denormalize(row[1])])

        to_output = pick_random(to_output, 60)
        writer.writerow(header)
        for row in to_output:
            writer.writerow(row)


if __name__ == "__main__":
    main()
