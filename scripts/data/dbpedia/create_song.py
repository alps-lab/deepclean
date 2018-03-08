#!/usr/bin/env python
import csv
import numpy as np


IN_PATH = "/home/xinyang/Datasets/dataclean/tables/DBPedia/prototype/new_Song.csv"


def pick_random(ls, num):
    n = len(ls)
    indices = np.random.choice(np.arange(n), num, replace=False)

    return [ls[i] for i in indices]


def denormalize(s):
    if "(" in s:
        return s.split("(")[0].strip()
    else:
        return s


def to_year(ds):
    pass


def main():
    languages = ["English language", "Dutch language", "Chinese language",
                 "Japanese language", "France", "Spanish language"]

    with open(IN_PATH, encoding="latin-1") as fobj:
        reader = csv.reader(fobj)
        header = next(reader)

        rows = [row for row in reader if row[3] in languages]

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/400", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 3]

        to_output = []
        for row in rows:
            if (row[0] != "NULL" and row[1] != "NULL"
                    and not row[0].startswith("{") and not row[1].startswith("{")):
                to_output.append(row)

        to_output = pick_random(to_output, 50)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(denormalize(row[i]) for i in use_cols)

    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/401", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 4]

        to_output = []
        for row in rows:
            new_row = []
            if row[0] != "NULL" and row[1] != "NULL":
                new_row.append(denormalize(row[0]))
                if row[1].startswith("{"):
                    new_row.append("; ".join([denormalize(entry.strip()) for entry in row[1][1:-1].split("|")]))
                else:
                    new_row.append(denormalize(row[1]))
                if row[4] == "NULL":
                    continue
                try:
                    year = int(row[4].split("-")[0])
                except ValueError:
                    continue

                if year >= 1940:
                    new_row.append(year)
                    to_output.append(new_row)

        # print(to_output)

        to_output = pick_random(to_output, 25)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(row)


    with open("/home/xinyang/Datasets/dataclean/tables/DBPedia/base/402", "w") as fobj:
        writer = csv.writer(fobj)
        use_cols = [0, 1, 2, 3]

        to_output = []
        for row in rows:
            new_row = [denormalize(row[0]), denormalize(row[1])]
            if (row[0] != "NULL" and row[1] != "NULL"
                    and not row[0].startswith("{") and not row[1].startswith("{")
                    and row[2] != "NULL"
                ):
                if row[2].startswith("{"):
                    new_row.append("; ".join([denormalize(entry.strip()) for entry in row[2][1:-1].split("|")[:2]]))
                else:
                    new_row.append(row[2])

                new_row.append(row[3])
                to_output.append(new_row)

        to_output = pick_random(to_output, 50)
        writer.writerow([header[i] for i in use_cols])
        for row in to_output:
            writer.writerow(row)


if __name__ == "__main__":
    main()
