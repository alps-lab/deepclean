#!/usr/bin/env python
from six.moves import cPickle

from tqdm import tqdm

from theclean.utils.wiki_utils import get_category_graph
from theclean.utils.wiki_utils import get_category_docs

OUTPUT_PATH = "/home/xinyang/Datasets/Wikipedia/category_support.pkl"


def dfs(graph, u, support, visited):
    visited.add(u)
    current_set = set()
    for _, v in graph.out_edges:
        if v in visited:
            try:
                current_set = current_set | support[v]
            except KeyError as ex:
                print("Error with key=%s" % ex.args[0])
        else:
            dfs(graph, v, support, visited)

    docs = get_category_docs(u)
    current_set.update(docs)
    support[u] = current_set


def dfs_along_the_graph(graph):
    visited = set()
    support = {}

    bar = tqdm(desc="# items")
    for u in graph.node:
        if u not in visited:
            dfs(graph, u, support, visited)
        bar.update(len(support))

    return support


if __name__ == "__main__":
    graph = get_category_graph()
    support = dfs_along_the_graph(graph)
    support = {key: len(value) for key, value in support.items()}

    with open(OUTPUT_PATH, "wb") as fobj:
        cPickle.dump(support, fobj, protocol=3)


