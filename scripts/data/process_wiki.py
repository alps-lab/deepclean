#!/usr/bin/env python
from __future__ import print_function, absolute_import

from collections import Counter, OrderedDict
from pathlib import Path
import csv

import ujson
from tqdm import tqdm

from lxml import etree

WIKI_XML_PATH = "/home/xinyang/Datasets/Wikipedia/enwiki-20180101-pages-articles-multistream.xml"
GOOD_TAGS = ["{http://www.mediawiki.org/xml/export-0.10/}page"]
WIKI_DATA_DIR = "/home/xinyang/Datasets/Wikipedia/extracted_as_drqa"


class TagTargets(object):

    def __init__(self):
        self.counter = Counter()

    def end(self, tag):
        self.counter[tag] += 1

    def close(self):
        return self.counter.most_common(300)


def test1():
    """
    Try to understand the structure of Wikipedia XML dumps
    :return:
    """
    parser = etree.XMLParser(target=TagTargets())
    results = etree.parse(WIKI_XML_PATH, parser)

    results = [OrderedDict(tag=tag, count=count)
               for tag, count in results]

    with open("samples/wikipedia_dump_common_tags.json", "w",
              encoding="utf8") as fobj:
        ujson.dump(results, fobj)


def test2():
    """
    Try to list all id, titles of Wikipedia articles
    :return:
    """
    paths = (list(Path(WIKI_DATA_DIR).glob("./*/wiki_*")) +
             list(Path(WIKI_DATA_DIR).glob("./*/wiki_*.bz2")))
    output_path = "./samples/wikipedia_dump_id_titles.csv"
    id_titles = []
    for path in tqdm(paths):
        with path.open() as fobj:
            for line in fobj:
                jobj = ujson.loads(line)
                id_titles.append([jobj["id"], jobj["title"]])

    with open(output_path, "w") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["id", "title"])

        for row in id_titles:
            writer.writerow(row)


if __name__ == "__main__":
    # test1()
    test2()
