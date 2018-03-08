#!/usr/bin/env python
import csv


CURRENCY_IN_PATH = "/Users/zhangxinyang/Downloads/DBPedia/Currency.csv"
HISTORIC_BUILDING_IN_PATH = "/Users/zhangxinyang/Downloads/DBPedia/HistoricBuilding.csv"

CURRENCY_OUT_PATH = "/Users/zhangxinyang/Downloads/DBPedia/new_Currency.csv"
HISTORIC_BUILDING_OUT_PATH = "/Users/zhangxinyang/Downloads/DBPedia/new_HistoricBuilding.csv"

SONG_IN_PATH = "/Users/zhangxinyang/Downloads/DBPedia/Song.csv"
SONG_OUT_PATH = "/Users/zhangxinyang/Downloads/DBPedia/new_Song.csv"

def do_currency():
    use_cols = [1, 2]
    with open(CURRENCY_IN_PATH) as fobj:
        reader = csv.reader(fobj)
        rows = [row for row in reader]
        header, rows = rows[0], rows[4:]

        header = [header[i] for i in use_cols]
        rows = [[row[i] for i in use_cols] for row in rows]

    with open(CURRENCY_OUT_PATH, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)


def do_historic_building():
    use_cols = [1, 4, 6, 15]
    with open(HISTORIC_BUILDING_IN_PATH) as fobj:
        reader = csv.reader(fobj)
        rows = [row for row in reader]
        header, rows = rows[0], rows[4:]

        header = [header[i] for i in use_cols]
        rows = [[row[i] for i in use_cols] for row in rows]

    with open(HISTORIC_BUILDING_OUT_PATH, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)


def do_song():
    use_cols = [2, 4, 8, 11, 25]
    with open(SONG_IN_PATH) as fobj:
        reader = csv.reader(fobj)
        rows = [row for row in reader]
        header, rows = rows[0], rows[4:]

        header = [header[i] for i in use_cols]
        rows = [[row[i] for i in use_cols] for row in rows]

    with open(SONG_OUT_PATH, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    do_currency()
    do_historic_building()
    do_song()

