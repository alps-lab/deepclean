#!/usr/bin/env python
import csv
from pathlib import Path
import argparse


relationGT_fields = {
    "0": ["Country", "Capital"],
    "1": ["State", "Capital"],
    "10": ["Year", "Author", "Novel"],
    "100": ["Actor","Movie"],
    "101": ["Actor", "Movie"],
    "13": ["Film Name", "Country", "Director", "Year"],
    "14": ["STATE", "CAPITAL CITY", "STATE NICKNAME"],
    "140": ["Actor", "Movie"],
    "17": ["Novel", "Author"],
    "18": ["Country", "Capital City"],
    "19": ["State", "Capital", "Largest city"],
    "2": ["Movie", "Actor"],
    "203": ["Actor", "Movie"],
    "209": ["Actor", "Movie"],
    "21": ["Title of Great Novel", "Author"],
    "210": ["Actor", "Movie"],
    "26": ["Actor", "Movie", "Type"],
    "28": ["Country", "Official Language"],
    "29": ["Year", "Actor", "Movie"],
    "3": ["Country", "Capital"],
    "300": ["Director", "Movie"],
    "4": ["President", "Actor", "Film", "Year"],
    "5": ["Actor", "President", "Movie"],
    "6": ["Actor", "President", "Movie"],
    "7": ["Year", "Film",  "Director", "Language"],
    "8": ["Year", "Director (Winner)", "Film", "Language"]
}

wikitables_fields = {
    "100": ["Actor", "Movie"],
    "11": ["Year", "Author", "Book"],
    "18": ["Year", "Name", "Author"],
    "23": ["Film", "Year", "Role"], #
    "25": ["President", "Library name", "Location", "Operated By"],
    "27": ["Prime Minister", "Date of Birth", "First mandate begins"],
    "28": ["Release date", "Product name", "Last IE"],
    "29": ["Operator", "Location", "Reactor", "Operational"], #
    "30": ["Park Name", "County or Counties"],
    "36": ["Film", "Year", "Score composer", "Title song", "Composed by", "Performed by"],
    "38": ["Original title (top) Alternative title (bottom)", "Directed by", "Written by"
            ,"Original airdate"],
    "4": ["Name", "Field", "Year"],
    "400": ["Composer", "Movie"], #
    "5": ["Car", "Film"],
    "6": ["Film", "Date of original release"],
    "8": ["Year", "Winner", "Film"],
    "9": ["Year", "Illustrator", "Book"],
}


def do_relationGT(source_dir, target_dir):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(exist_ok=True)

    for path in Path(source_dir).iterdir():
        name = path.name
        if name not in relationGT_fields:
            continue
        with open(str(path)) as fobj:
            reader = csv.reader(fobj)
            header = next(reader)

            good_cols = []
            for i, token in enumerate(header):
                if token in relationGT_fields[name]:
                    good_cols.append(i)

            assert all(field in header for field in relationGT_fields[name]), "field name error on %s" % str(path)
            rows = []
            for row in reader:
                if all(len(token.strip()) > 0 for token, field in zip(row, header)
                       if field in relationGT_fields[name]):
                    rows.append(row)

        with target_dir_path.joinpath(name).open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow([header[i] for i in good_cols])

            for row in rows:
                writer.writerow([row[i] for i in good_cols])


def do_wikitables(source_dir, target_dir):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(exist_ok=True)

    for path in Path(source_dir).iterdir():
        name = path.name
        if name not in wikitables_fields:
            continue
        with open(str(path)) as fobj:
            reader = csv.reader(fobj)
            header = next(reader)

            good_cols = []
            for i, token in enumerate(header):
                if token in wikitables_fields[name]:
                    good_cols.append(i)

            assert all(field in header for field in wikitables_fields[name]), "field name error on %s" % str(path)
            rows = []
            for row in reader:
                if all(len(token.strip()) > 0 for token, field in zip(row, header)
                       if field in wikitables_fields[name]):
                    rows.append(row)

        target_path = target_dir_path.joinpath(name)
        if target_path.exists():
            continue

        with target_path.open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow([header[i] for i in good_cols])

            for row in rows:
                writer.writerow([row[i] for i in good_cols])


def do_dbpedia(source_dir, target_dir):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(exist_ok=True)

    for path in Path(source_dir).iterdir():
        name = path.name

        with open(str(path)) as fobj:
            reader = csv.reader(fobj)
            header = next(reader)

            rows = []
            for row in reader:
                rows.append(row)

        target_path = target_dir_path.joinpath(name)
        if target_path.exists():
            continue

        with target_path.open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow(header)

            for row in rows:
                writer.writerow(row)


def do_l2(source_dir, target_dir):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(exist_ok=True)

    for path in Path(source_dir).iterdir():
        name = path.name

        with open(str(path)) as fobj:
            reader = csv.reader(fobj)
            header = next(reader)

            rows = []
            for row in reader:
                rows.append(row)

        target_path = target_dir_path.joinpath(name)
        if target_path.exists():
            continue

        with target_path.open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow(header)

            for row in rows:
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", metavar="WHICH", choices=["relationGT", "wikitables",
                                                           "DBPedia", "l2"])
    parser.add_argument("source_dir", metavar="SOURCE_DIR")
    parser.add_argument("target_dir", metavar="TARGET_DIR")

    config = parser.parse_args()

    if config.which == "relationGT":
        do_relationGT(config.source_dir, config.target_dir)
    elif config.which == "wikitables":
        do_wikitables(config.source_dir, config.target_dir)
    elif config.which == "DBPedia":
        do_dbpedia(config.source_dir, config.target_dir)
    else:
        do_l2(config.source_dir, config.target_dir)


if __name__ == "__main__":
    main()