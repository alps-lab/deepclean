#!/usr/bin/env python
from six.moves import cPickle, xrange

import sys
import argparse
import csv
from pathlib import Path
import heapq

from tqdm import tqdm

from theclean.experimental.prototype2.actions import (annotate_documents,
                    mark_top_doc_scores_categories, annotate_entities, mark_literal_ner,
                    recalculate_graph_categories, search_question_template, assign_documents,
                    mark_category_scores, do_query, search_template_for_columns)
from theclean.utils.data_utils import read_dataset
from theclean.utils.drqa_utils import RANK_SERVER
from theclean.utils.data_elements import EEntry, TEntry
from theclean.utils.data_utils import chunk_iter



def read_csv(path):
    with open(path) as fobj:
        reader = csv.reader(fobj)
        header = next(reader)
        rows = [[entry for entry in row] for row in reader]

        return read_dataset(header, rows)


def do_annotate(config):
    Path(config.target_dir).mkdir(exist_ok=True)

    for path in tqdm(list(Path(config.source_dir).iterdir())):
        name = path.name
        target_path = Path(config.target_dir).joinpath(name + ".pkl")
        if target_path.exists():
            continue

        dataset = read_csv(str(path))
        try:
            for row in dataset:
                mark_top_doc_scores_categories(row, rank_server=RANK_SERVER,
                                               name="doc_score_categories", k=1)
            mark_category_scores(dataset)
            for row in dataset:
                mark_top_doc_scores_categories(row, rank_server="http://localhost:4444",
                                                       name="doc_score_categories", k=30)

            name_score = 2.0
            for it in xrange(4):
                assign_documents(dataset, name_score=name_score)
                recalculate_graph_categories(dataset)
                name_score *= 0.5

            mark_literal_ner(dataset)

        except Exception as ex:
            print("failed at %s" % str(path), file=sys.stderr)
            raise ex
        else:
            with open(str(target_path), "wb") as fobj:
                cPickle.dump(dataset, fobj, protocol=3)


def do_search_templates(config):
    Path(config.target_dir).mkdir(exist_ok=True)

    for path in tqdm(list(Path(config.source_dir).iterdir())):
        name = path.name
        target_path = Path(config.target_dir).joinpath(name)
        if target_path.exists():
            continue

        with open(str(path), "rb") as fobj:
            dataset = cPickle.load(fobj)

        num_rows, num_cols = len(dataset), len(dataset[0])
        if num_cols < 3:
            total_steps = 1
            ks = [15]
        elif num_cols >= 3:
            total_steps = 2
            ks = [25, 15]
        else:
            raise ValueError("number of column too small: %d" % num_cols)

        try:
            search_question_template(dataset, total_steps=total_steps,
                                    external=False, ks=ks)
            for j in xrange(len(dataset[0])):
                if (len(dataset.attrs["candidate_templates"][j]) == 0 or
                        dataset.attrs["candidate_templates"][j][0].score < 0.3):
                    old_states = dataset.attrs["candidate_templates"][j]
                    other_states = search_template_for_columns(dataset, j,
                                            total_steps, external=True, ks=ks)
                    all_states = list(heapq.merge(old_states, other_states, reverse=True))[:10]
                    dataset.attrs["candidate_templates"][j] = all_states

        except Exception as ex:
            print("failed at %s" % str(path), file=sys.stderr)
            raise ex
        else:
            with open(str(target_path), "wb") as fobj:
                cPickle.dump(dataset, fobj, protocol=3)


def _do_fill(dataset, name="best_fill", removed_doc_for_current=True,
             entry_types=None, top_n=5):
    if entry_types is None:
        entry_types = (EEntry, )
    else:
        entry_types = tuple(entry_types)

    doc_names_by_row = []
    for row in dataset:
        doc_names_by_row.append([])
        for j, entry in enumerate(row):
            if "best_candidate_doc" in entry.attrs:
                if (j not in dataset.attrs["ner_annotation"] or
                            dataset.attrs["ner_annotation"][j] not in ["DATE", "TIME", "NUMBER",
                                                                       "MONEY", "PERCENT",
                                                                       "DURATION"]):
                    doc_names_by_row[-1].append(entry["best_candidate_doc"])

    question_entries, doc_names, doc_scores = [], [], []
    for i, row in enumerate(dataset):
        for j, entry in enumerate(row):
            if j in dataset.attrs["candidate_templates"] and len(dataset.attrs["candidate_templates"]) > 0:
                templates, column_phrases = dataset.attrs["candidate_templates"][j], dataset.attrs["column_phrases"]

                if len(dataset.attrs["candidate_templates"][j]) == 0:
                    continue

                if isinstance(entry, entry_types):
                    question = ["Which", "is"]

                    if not dataset.attrs["candidate_templates"][j][0].external:
                        doc_names.append(doc_names_by_row[i][:])
                        if removed_doc_for_current and "best_candidate_doc" in entry.attrs:
                            if doc_names[-1].count(entry["best_candidate_doc"]) == 1:
                                del doc_names[-1][doc_names[-1].index(entry["best_candidate_doc"])]
                        doc_scores.append([1.0 for _ in xrange(len(doc_names[-1]))])
                    else:
                        doc_names.append([])
                        doc_scores.append([])

                    for t, (u, v) in enumerate(templates[0].components):
                        if t == 0:
                            question.append(column_phrases[u][v])
                            question.append("for")
                        if t > 1:
                            question.append("and")
                        if t > 0:
                            question.append("" if isinstance(row[u], EEntry) else row[u].text)

                    question_entries.append((" ".join(question), (i, j)))

    for sub_question_entries, sub_doc_names, sub_doc_scores \
            in zip(chunk_iter(question_entries, 512),
                   chunk_iter(doc_names, 512), chunk_iter(doc_scores, 512)):
        # print(len(doc_names), len(doc_scores))
        result_obj = do_query([question for question, _ in sub_question_entries], top_n=top_n, n_docs=5,
                              doc_names=sub_doc_names, doc_scores=sub_doc_scores)
        for question_entry, result in zip(sub_question_entries, result_obj):
            _, (i, j) = question_entry
            best_fills = []
            for sub_robj in result:
                best_fills.append(sub_robj["span"])

            dataset[i][j].attrs[name] = best_fills


def do_fill(config):
    Path(config.target_dir).mkdir(exist_ok=True)

    for path in tqdm(list(Path(config.source_dir).iterdir())):
        name = path.name
        target_path = Path(config.target_dir).joinpath(name)
        if target_path.exists():
            continue

        with open(str(path), "rb") as fobj:
            dataset = cPickle.load(fobj)

        _do_fill(dataset, removed_doc_for_current=False, top_n=config.top_n)

        with open(str(target_path), "wb") as fobj:
            cPickle.dump(dataset, fobj)


def validate_and_fill(config):
    Path(config.target_dir).mkdir(exist_ok=True)

    for path in tqdm(list(Path(config.source_dir).iterdir())):
        name = path.name
        target_path = Path(config.target_dir).joinpath(name)
        if target_path.exists():
            continue

        with open(str(path), "rb") as fobj:
            dataset = cPickle.load(fobj)

        _do_fill(dataset, "correction_fills", True, entry_types=[TEntry])
        # _do_fill(dataset, "correction_fills", True, entry_types=[TEntry])

        with open(str(target_path), "wb") as fobj:
            cPickle.dump(dataset, fobj)


def get_action_func(s):
    if s == "annotate":
        return do_annotate
    if s == "search_templates":
        return do_search_templates
    if s == "fill":
        return do_fill
    if s == "validate_and_fill":
        return validate_and_fill

    raise ValueError("Unsupported action %s" % s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", metavar="SOURCE_DIR")
    parser.add_argument("target_dir", metavar="TARGET_DIR")
    parser.add_argument("action", metavar="ACTION", choices=["annotate", "search_templates"
                                                             , "fill", "corrupt",
                                                             "validate_and_fill"])
    parser.add_argument("--top-n", dest="top_n",
                        default=5, type=int)

    config = parser.parse_args()
    assert Path(config.source_dir) != Path(config.target_dir)
    func = get_action_func(config.action)
    func(config)


if __name__ == "__main__":
    main()

