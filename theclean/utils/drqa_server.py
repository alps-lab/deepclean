#!/usr/bin/env python
import argparse
import logging
from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler
import traceback

import ujson
import numpy as np


RANKER_MODEL_PATH = "/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa-tfidf-ngram=2-hash=16777216-tokenizer=corenlpserver.npz"
READER_MODEL_PATH = "/home/xinyang/.drqa_data/reader/multitask.mdl"
RANKER_CONFIG = {'options': {'tfidf_path': RANKER_MODEL_PATH, 'strict': False}}
DB_PATH = "/home/xinyang/Datasets/Wikipedia/wikipedid_20180101_drqa.db"
DB_CONFIG = {'options': {'db_path': DB_PATH}}


class Handler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super(Handler, self).__init__(*args, **kwargs)

    def _set_header(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_POST(self):
        data_string = self.rfile.read(int(self.headers["Content-Length"])).decode()
        try:
            data = ujson.loads(data_string)
            queries = data["questions"]
            doc_names, doc_scores = data["doc_names"], data["doc_scores"]
            top_n = data.get("top_n", 1)
            n_docs = data.get("n_docs", 5)
            return_context = data.get("return_context", False)
            exclusive = data.get("exclusive", True)

            if len(doc_names) == 0:
                candidate_doc_scores = None
            else:
                candidate_doc_scores = [(sub_doc_names, np.asarray(sub_doc_scores, np.float64)) for sub_doc_names, sub_doc_scores
                                        in zip(doc_names, doc_scores)]

            result_obj = self.server.drqa.process_batch(queries, candidate_doc_scores=candidate_doc_scores,
                                                        return_context=return_context, n_docs=n_docs, top_n=top_n,
                                                        )

        except Exception as ex:
            traceback.print_exc()
            self.send_error(400)
        else:
            self._set_header()
            self.wfile.write(ujson.dumps(result_obj).encode())


def init_server(server, cuda):
    from theclean.utils.drqa.drqa_pipeline import DrQA

    drqa = DrQA(READER_MODEL_PATH, tokenizer="corenlpserver", cuda=cuda,
                ranker_config=RANKER_CONFIG, db_config=DB_CONFIG)
    server.drqa = drqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", dest="ip", default="0.0.0.0")
    parser.add_argument("-p", "--port", dest="port", default=5555, type=int)
    parser.add_argument("-c", "--cuda", dest="cuda", action="store_true")

    config = parser.parse_args()
    server = HTTPServer((config.ip, config.port), Handler)

    logging.getLogger().setLevel(logging.INFO)

    logging.log(logging.INFO, "initializing the server...")
    init_server(server, config.cuda)
    logging.log(logging.INFO, "initialized.")
    try:
        server.serve_forever(poll_interval=1)
    except KeyboardInterrupt:
        logging.log(logging.INFO, "interrupted...")
        server.server_close()


