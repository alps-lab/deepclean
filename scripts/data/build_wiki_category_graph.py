#!/usr/bin/env python
import csv
import sys

from tqdm import tqdm
import networkx as nx
import matplotlib
matplotlib.use("Agg") # set backend for Matplotlib
import matplotlib.pyplot as plt

from theclean.utils.wiki_utils import category_filter

DB_PATH = "/home/xinyang/Datasets/Wikipedia/enwiki_leveldb"
CATEGORYLINKS_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-categorylinks-clean.csv"
PAGE_PATH = "/home/xinyang/Datasets/Wikipedia/csv/enwiki-20180101-page-clean.csv"
SAVE_PATH = "/home/xinyang/Datasets/Wikipedia/category-graph.pkl"


def main():
    graph = nx.DiGraph()

    pageid_catename_map = {}
    with open(PAGE_PATH) as fobj:
        reader = csv.reader(fobj)
        for page_id, ns_id, page_name in tqdm(reader):
            ns_id = int(ns_id)
            if ns_id != 14:
                continue
            pageid_catename_map[page_id] = page_name
            if category_filter(page_name):
                graph.add_node(page_name)

    bad_keys = set()
    with open(CATEGORYLINKS_PATH, encoding="latin1") as fobj:
        reader = csv.reader(fobj)
        for row in tqdm(reader):
            if row[2] != "subcat":
                continue

            try:
                cate_name, super_catename = pageid_catename_map[row[0]], row[1]
                if category_filter(cate_name) and category_filter(super_catename):
                    graph.add_edge(super_catename, cate_name)
            except KeyError as ex:
                bad_keys.add(ex.args[0])

    print("the following keys are failed", file=sys.stderr)
    print(", ".join(bad_keys), file=sys.stderr)

    nx.write_gpickle(graph, SAVE_PATH, protocol=3)


if __name__ == "__main__":
    main()