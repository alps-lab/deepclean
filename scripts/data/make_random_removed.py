#!/usr/bin/env python
from six.moves import xrange

from pathlib import Path
import argparse
import csv

import numpy as np


def main(config):
    target_dir_path = Path(config.target_dir)
    assert target_dir_path != Path(config.source_dir)
    target_dir_path.mkdir(exist_ok=True)

    rs = np.random.RandomState(config.seed)
    for path in sorted(Path(config.source_dir).iterdir()):
        target_path = target_dir_path.joinpath(path.name)

        with path.open() as fobj:
            reader = csv.reader(fobj)
            header = next(reader)
            rows = [row for row in reader]

        with target_path.open("w") as fobj:
            writer = csv.writer(fobj)
            writer.writerow(header)

            for row in rows:
                for i in xrange(len(row)):
                    if rs.uniform() < config.ratio:
                        row[i] = ""
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", metavar="SOURCE_DIR")
    parser.add_argument("target_dir", metavar="TARGET_DIR")
    parser.add_argument("-r", "--ratio", dest="ratio", type=float, default=0.1)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=777999444)

    main(parser.parse_args())