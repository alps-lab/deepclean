from six.moves import cPickle

import warnings
import sqlite3
import ujson
from collections import Counter
from functools import lru_cache

import networkx as nx
import plyvel


DATABASE_PATH = "/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa.db"
LEVELDB_PATH = "/home/xinyang/Datasets/Wikipedia/enwiki_leveldb"
CATEGORY_GRAPH_PATH = "/home/xinyang/Datasets/Wikipedia/category-graph.pkl"
CATEGORY_SUPPORTS_PATH = "/home/xinyang/Datasets/Wikipedia/category_support.pkl"

_conn_doc, _conn_leveldb = None, None
NUM_WIKI_DOCS = 5295907

_category_graph, _category_supports = None, None


BAD_SET = ("Wikipedia_articles_", "Pages_", "All_articles_", "Articles_",
                        "Webarchive_", "Wikipedia_", "Lists_", "All_Wikipedia_", "Featured_articles",
                        "All_Wikipedia_", "Good_articles")


def get_conn_doc():
    global _conn_doc
    if _conn_doc is None:
        _conn_doc = sqlite3.connect(DATABASE_PATH)
    return _conn_doc


def get_conn_leveldb():
    global _conn_leveldb
    if _conn_leveldb is None:
        _conn_leveldb = plyvel.DB(LEVELDB_PATH)
    return _conn_leveldb


def read_doc(title=None, wikiid=None):
    conn, cursor = get_conn_doc(), None
    try:
        if title is not None:
            cursor = conn.execute("SELECT * FROM documents "
                                  "WHERE id=?", (title,))
        else:
            cursor = conn.execute("SELECT * FROM documents "
                                  "WHERE wikiid=?", (wikiid,))
        return cursor.fetchall()
    except Exception:
        raise
    finally:
        if cursor:
            cursor.close()


def read_doc_categories(title=None, wikiid=None):
    if title is not None:
        db = get_conn_leveldb()
        wikiid = db.get(b"title_id_" + title.encode())

    assert wikiid is not None
    if isinstance(wikiid, str):
        wikiid = wikiid.encode()

    return list(filter(category_filter, ujson.loads(db.get(b"id_category_" + wikiid).decode())))


def category_filter(name):
    if name.startswith(BAD_SET):
        return False

    return True


def count_category_along_paths(graph, counter):
    bfs_counter = Counter()

    for node, count in counter.items():
        if node in graph:
            bfs_counter[node] += count
            for u, v in nx.bfs_edges(graph, node, reverse=True):
                bfs_counter[v] += count

    return bfs_counter


def prune_graph_with_tops(graph, tops):
    candidate_freq = {}
    for top_cate in tops:
        if top_cate in graph:
            candidate_freq[top_cate] = get_category_frequency(top_cate)

            for _, v in nx.bfs_edges(graph, top_cate):
                candidate_freq[v] = get_category_frequency(v)

            for u, v in nx.bfs_edges(graph, top_cate, reverse=True):
                candidate_freq[v] = get_category_frequency(v) + candidate_freq[u]

    return candidate_freq


def get_category_graph():
    global _category_graph
    if _category_graph is None:
        _category_graph = nx.read_gpickle(CATEGORY_GRAPH_PATH)
    return _category_graph


def get_category_docs(name):
    db = get_conn_leveldb()
    obj = db.get(b"category_id_" + name.encode())
    if obj is None:
        return []
    else:
        return ujson.loads(obj.decode())


def get_category_supports():
    global _category_supports
    if _category_supports is None:
        with open(CATEGORY_SUPPORTS_PATH, "rb") as fobj:
            _category_supports = cPickle.loads(fobj)
    return _category_supports


@lru_cache(1024)
def get_category_frequency(name):
    db = get_conn_leveldb()
    obj = db.get(b"category_id_" + name.encode())
    if obj is None:
        # warnings.warn("Key %s not found." % name)
        return 0

    return len(ujson.loads(obj.decode()))
