import drqa.retriever


RETRIEVER_MODEL = "/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa-tfidf-ngram=2-hash=16777216-tokenizer=corenlpserver.npz"

_ranker = None

RANK_SERVER = "http://localhost:4444"


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = drqa.retriever.get_class("tfidf")(tfidf_path=RETRIEVER_MODEL)
    return _ranker
