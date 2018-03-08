#!/usr/bin/env python
from six.moves import xrange

from pathlib import Path
import argparse
import csv

import numpy as np


def main(config):
    target_dir_path = Path(config.target_dir)
    correct_dir_path = Path(config.correct_dir)
    assert target_dir_path != Path(config.source_dir)
    target_dir_path.mkdir(exist_ok=True)
    correct_dir_path.mkdir(exist_ok=True)

    rs = np.random.RandomState(config.seed)
    for path in sorted(Path(config.source_dir).iterdir()):
        target_path = target_dir_path.joinpath(path.name)
        target_path_correct = correct_dir_path.joinpath(path.name + ".correct")

        with path.open() as fobj:
            reader = csv.reader(fobj)
            header = next(reader)
            rows = [row for row in reader]

        assert all([len(row) == len(header)] for row in rows), \
            "the data cannot be presented in a matrix."

        indices = rs.permutation(np.arange(0, len(rows), dtype=np.int64))
        num_rows_as_vocab = int(0.1 * len(rows))
        num_rows_to_corrupt = len(rows) - num_rows_as_vocab

        rows_to_corrupt = [rows[idx] for idx in indices[:num_rows_to_corrupt]]
        rows_as_vocabs = [rows[idx] for idx in indices[-num_rows_to_corrupt:]]
        vocab = {i: [] for i in xrange(len(header))}

        for row in rows_as_vocabs:
            for j, item in enumerate(row):
                vocab[j].append(item)

        corrupted_rows, correct_rows = [], []

        for row in rows_to_corrupt:
            row = row[:]
            correct_rows.append(row[:])
            for j in xrange(len(row)):
                if np.random.uniform() < config.ratio:
                    row[j] = np.random.choice(vocab[j])
            corrupted_rows.append(row)

        # target_path = target_dir_path.joinpath(name)
        # if target_path.exists():
        #     continue

        if target_path.exists():
            continue

        with target_path.open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow(header)

            for row in corrupted_rows:
                writer.writerow(row)

        with target_path_correct.open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow(header)

            for row in correct_rows:
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", metavar="SOURCE_DIR")
    parser.add_argument("target_dir", metavar="TARGET_DIR")
    parser.add_argument("correct_dir", metavar="CORRECT_DIR")
    parser.add_argument("-r", "--ratio", dest="ratio", type=float, default=0.1)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=777999444)

    main(parser.parse_args())