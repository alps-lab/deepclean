#!/usr/bin/env python
import traceback
import argparse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import ujson
from drqa.retriever import get_class
from theclean.utils.drqa_utils import RETRIEVER_MODEL


class Handler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwarsg):
        super(Handler, self).__init__(*args, **kwarsg)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        self.data_string = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            data = ujson.loads(self.data_string)
            query, k = data["query"], data.get("k", 1)
            ranker = self.server.handler_params["ranker"]
            doc_names, doc_scores = ranker.closest_docs(query, k)

            ret_obj = {"doc_names": [name for name in doc_names],
                       "doc_scores": ["%.2f" % score for score in doc_scores]}

        except Exception as ex:
            self.send_error(400)
            traceback.print_exc()

        else:
            self._set_headers()
            self.wfile.write(ujson.dumps(ret_obj).encode())

    def do_GET(self):
        self.send_error(404)


def configurate_server(server, tfidf_path):
    server.handler_params = {"ranker": get_class("tfidf")(tfidf_path=tfidf_path, strict=False)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest="model_path", default=RETRIEVER_MODEL)
    parser.add_argument("--ip", dest="ip", default="localhost")
    parser.add_argument("--port", dest="port", type=int, default=4444)

    config = parser.parse_args()

    server = None
    try:
        print("Starting...", file=sys.stderr)
        server = HTTPServer((config.ip, config.port), Handler)
        configurate_server(server, config.model_path)
        print("Started...", file=sys.stderr)
        server.serve_forever()

    except KeyboardInterrupt:
        print("Interrupted...", file=sys.stderr)
        server.shutdown()

    finally:
        if server:
            server.shutdown()
