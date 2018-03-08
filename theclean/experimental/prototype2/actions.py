#!/usr/bin/env python
from six.moves import xrange

import warnings
from itertools import combinations, product
from collections import Counter, defaultdict
import multiprocessing as mp
import re
from difflib import SequenceMatcher
import random

import ujson
import requests
import numpy as np
from nltk import FreqDist
from sklearn.metrics import f1_score, recall_score

from theclean.utils.wiki_utils import (read_doc, read_doc_categories, category_filter,
                                       NUM_WIKI_DOCS, get_category_frequency, get_category_graph,
                                       prune_graph_with_tops, count_category_along_paths)
from theclean.utils.text_utils import (
    annotate, extract_sentences, NLP_PROPERTIES_COREF, NLP_PROPERTIES_IE, NLP_SERVER,
    mark_tokens, STOPWORDS, extract_noun_phrases, NLP_PROPERTIES_LEMMA, NLP_PROPERTIES_NER)
from theclean.utils.data_elements import TEntry, TextSpan, BinaryRelationS, to_compact_relation
from theclean.experimental.prototype2.search import BeamSearch, State


PARENTHESIS_PATTERN = re.compile(r"-LRB-\s+.*?\s+-RRB-")

NER_MAP = {
    "LOCATION": ["place", "region", "country"],
    "PERSON": ["person"],
    "DATE": ["year", "day", "date"],
}


def split_title(title):
    if "(" in title:
        title = title.split("(")[0].strip()

    return title


def remove_parenthesis(text):
    return " ".join(PARENTHESIS_PATTERN.split(text)).strip()


def remove_parenthesis_on_tokens(tokens):
    index_set = set()
    skip_mode = False
    stack = []
    for i, token in enumerate(tokens):
        if token == '-LRB-':
            skip_mode = True
            stack.append(token)
        elif token == '-RRB-':
            if len(stack) > 0:
                stack.pop()
            if len(stack) == 0:
                skip_mode = False
        elif not skip_mode:
            index_set.add(i)

    return index_set


def remove_parenthesis_on_tokens_text(tokens):
    index_set = sorted(remove_parenthesis_on_tokens(tokens))
    return " ".join(tokens[idx] for idx in index_set)


def mark_top_doc_scores_categories(row, rank_server, name="doc_score_categories", k=3,
                        ):
    for entry in row:
        if isinstance(entry, TEntry) and not entry.text.isdigit(): # to remove digital literals
            text = entry.text
            query_obj = dict(query=text, k=k)
            req = requests.post(rank_server, json=query_obj)
            robj = req.json()

            doc_names, doc_scores = robj["doc_names"], robj["doc_scores"]
            if len(doc_names) == 0:
                continue
            doc_scores = [float(score) for score in doc_scores]
            doc_categories = [list(filter(category_filter, read_doc_categories(doc_name)))
                              for doc_name in doc_names]

            entry[name] = [dict(doc_name=doc_name, doc_score=doc_score,
                                categories=doc_category)
                           for doc_name, doc_score, doc_category
                           in zip(doc_names, doc_scores, doc_categories)]


def mark_category_scores(dataset, name="column_category_scores"):
    num_rows, num_cols = len(dataset), len(dataset[0])
    dataset.attrs[name] = {}
    for j in xrange(num_cols):
        mark_cnt = 0
        for i in xrange(num_rows):
            try:
                item = dataset[i][j]
            except IndexError:
                print(i, j, dataset[i].entries)

            if "doc_score_categories" in item.attrs:
                mark_cnt += 1
        if mark_cnt < 0.2 * num_rows:
            continue
        counter = Counter()

        for i in xrange(num_rows):
            item = dataset[i][j]
            if "doc_score_categories" in item.attrs:
                subset = set(category for elem in item["doc_score_categories"]
                             for category in elem["categories"])
                for elem in subset:
                    counter[elem] += np.log2(NUM_WIKI_DOCS / get_category_frequency(elem))

        dataset.attrs[name][j] = counter


def assign_documents(dataset, name="best_candidate_doc", name_score=1.0):
    num_rows, num_cols = len(dataset), len(dataset[0])
    for j in xrange(num_cols):
        if j not in dataset.attrs["column_category_scores"]:
            continue

        col_scores = dataset.attrs["column_category_scores"][j]
        cate_idx = {}

        normalized_bg = np.zeros((50,), np.float64)
        for i, (cat_name, score) in enumerate(col_scores.most_common(50)):
            cate_idx[cat_name] = i
            normalized_bg[i] = score

        for i in xrange(num_rows):
            item = dataset[i][j]
            if "doc_score_categories" in item.attrs:
                name_scores = []

                for elem in item["doc_score_categories"]:
                    doc_name = elem["doc_name"]
                    cats = elem["categories"]

                    normalized_fg = np.zeros((50,), np.float64)
                    for cat in cats:
                        if cat in cate_idx:
                            normalized_fg[cate_idx[cat]] = np.log2(NUM_WIKI_DOCS / get_category_frequency(cat))

                    val1 = np.log2(np.dot(normalized_fg, normalized_bg) + 2.0)
                    val2 = np.log2(elem["doc_score"] + 2.0)

                    name_scores.append((doc_name, dict(v1=val1, v2=val2)))

                maxv1 = max(ns[1]["v1"] for ns in name_scores)
                maxv2 = max(ns[1]["v2"] for ns in name_scores)

                best_doc, best_scores = -1, 0.0
                for doc_name, scores in name_scores:
                    scores["v1"] /= maxv1
                    scores["v2"] /= maxv2
                    scores["v2"] *= 1.25

                    if split_title(doc_name).lower() == item.text.lower().strip():
                        scores["v2"] += name_score

                    v = scores["v1"] + scores["v2"]
                    if v > best_scores:
                        best_doc = doc_name
                        best_scores = v

                if best_doc != -1:
                    item[name] = best_doc
                    item[name + "_ref"] = name_scores


def annotate_documents(dataset, name="annotated_doc"):
    ctx, pool = None, None
    doc_names = set()

    for row in dataset:
        for entry in row:
            if "best_candidate_doc" in entry.attrs:
                doc_names.add(entry["best_candidate_doc"])

    doc_names = sorted(doc_names)
    doc_name_index = {doc_name: idx for idx, doc_name in enumerate(doc_names)}
    doc_contents = ["\n\n".join(read_doc(doc_name)[0][1].split("\n\n")[1:]) for doc_name in doc_names]
    doc_contents = [(doc_name + ".\n\n" + doc_content)[:100000]
                    for doc_name, doc_content in zip(doc_names, doc_contents)]

    try:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(8)
        doc_annotations = pool.starmap(annotate, [(content, NLP_SERVER, NLP_PROPERTIES_NER)
                                      for content in doc_contents])
    finally:
        if pool:
            pool.close()

    for row in dataset:
        for entry in row:
            if "best_candidate_doc" in entry.attrs:
                annotated_obj = doc_annotations[doc_name_index[entry["best_candidate_doc"]]]
                if not isinstance(annotated_obj, str):
                    entry[name] = annotated_obj
                else:
                    warnings.warn(message="failed on document %s, with: %s"
                                                           "" % (entry["best_candidate_doc"], annotated_obj))


def annotate_entities(row, name="entity_ranges"):
    entity_ranges = {}
    entry_tokens, entry_titles = [], []

    for entry in row:
        if isinstance(entry, TEntry):
            tokens = [token["word"] for token in annotate(entry.text)["sentences"][0]["tokens"]]
            entry_tokens.append(tokens)
            entry_titles.append(entry.text)

    for entry in row:
        if "best_candidate_doc" in entry.attrs:
            doc_name = entry["best_candidate_doc"]
            annotated_doc = entry["annotated_doc"]
            doc_tokens = [token["word"] for sent in annotated_doc["sentences"]
                for token in sent["tokens"]]

            entity_ranges[doc_name] = {title: set() for title in entry_titles}
            for tokens, title in zip(entry_tokens, entry_titles):
                mark_ranges = mark_tokens(doc_tokens, tokens, lower=True)
                for mark_range in mark_ranges:
                    entity_ranges[doc_name][title].update(xrange(*mark_range))

            cumsum, sent_start = 0, []

            for sentence_obj in annotated_doc["sentences"]:
                sent_start.append(cumsum)
                cumsum += len(sentence_obj["tokens"])

            for chain in annotated_doc["corefs"].values():
                accepted = {title: False for title in entry_titles}
                to_add = {title: set() for title in entry_titles}

                for item in chain:
                    start_index = sent_start[item["sentNum"] - 1] + item["startIndex"] - 1
                    end_index = sent_start[item["sentNum"] - 1] + item["endIndex"] - 1

                    index_set = {start_index + idx for idx in
                                 remove_parenthesis_on_tokens(doc_tokens[start_index:end_index])}
                    # print(doc_tokens[start_index:end_index])
                    tokens = [doc_tokens[idx].lower() for idx in sorted(index_set)]

                    for title, title_tokens in zip(entry_titles, entry_tokens):
                        int_set = index_set & entity_ranges[doc_name][title]
                        to_add[title].update(index_set)
                        if len(int_set) > 0.25 * len(index_set):
                            accepted[title] = True
                        elif SequenceMatcher(None, tokens,
                            [token.lower() for token in title_tokens]).ratio() > 0.6:
                            accepted[title] = True
                        elif mark_tokens(tokens, title_tokens, lower=True):
                            accepted[title] = True

                for title in entry_titles:
                    # print(title, to_add[title], accepted[title])
                    if accepted[title]:
                        entity_ranges[doc_name][title].update(to_add[title])

    row.attrs[name] = entity_ranges


def pull_relations(row, name="relations"):
    relations = []

    for entry in row:
        if "annotated_doc" in entry.attrs:
            annotated_obj = entry["annotated_doc"]
            doc_name = entry["best_candidate_doc"]
            doc_tokens_with_tags = [token for sent in annotated_obj["sentences"] for token in sent["tokens"]]

            cumlen = 0
            for sent_num, sent_obj in enumerate(annotated_obj["sentences"]):
                for reln in sent_obj["openie"]:
                    sub_start_index, sub_end_index = reln["subjectSpan"]
                    obj_start_index, obj_end_index = reln["objectSpan"]

                    sub_start_index, sub_end_index = sub_start_index + cumlen, sub_end_index + cumlen
                    obj_start_index, obj_end_index = obj_start_index + cumlen, obj_end_index + cumlen

                    sub_indices = set(xrange(sub_start_index, sub_end_index))
                    obj_indices = set(xrange(obj_start_index, obj_end_index))

                    subjects, objects = [], []
                    accepted = False

                    for title, target_indices in row.attrs["entity_ranges"][doc_name].items():
                        int_indices = sub_indices & target_indices
                        if len(int_indices) > 0.25 * len(sub_indices):
                            accepted = True
                            subjects.append(title)
                        int_indices = obj_indices & target_indices
                        if len(int_indices) > 0.25 * len(obj_indices):
                            accepted = True
                            objects.append(title)

                    if accepted:
                        relations.append(
                            BinaryRelationS(reln["subject"], reln["object"],
                                    reln["relation"], subjects, objects,
                                    doc_name, sent_num, doc_tokens_with_tags[cumlen:cumlen+len(sent_obj["tokens"])],
                                    reln["subjectSpan"], reln["objectSpan"], reln["relationSpan"]))

                cumlen += len(sent_obj["tokens"])

    row.attrs[name] = relations


def extract_cross_relations(row, name="cross_relations"):
    cross_relations = []
    for relation in row.attrs["relations"]:
        if (len(relation.subjects) > 1 or len(relation.objects) > 1
            or (len(relation.subjects) == 1 and len(relation.objects) == 1 and
                relation.subjects[0] != relation.objects[0])):
            cross_relations.append(relation)

    row.attrs[name] = cross_relations


def mark_relations(row, name="relations"):
    pull_relations(row)
    extract_cross_relations(row)


def assign_relations(dataset, name="cross_relations"):
    num_rows, num_cols = len(dataset), len(dataset[0])
    cross_relations = {}
    for u, v in product(xrange(num_cols), repeat=2):
        if u == v:
            continue

        cross_sents = []
        deduplicated_set = set()

        for i, row in enumerate(dataset):
            if isinstance(row[u], TEntry) and isinstance(row[v], TEntry):
                for reln in row.attrs["cross_relations"]:
                    if (row[u].text.lower() in set(obj.lower() for obj in reln.subjects)
                            and row[v].text.lower() in set(obj.lower() for obj in reln.objects)):
                        if (reln.doc_name, reln.sent_num) in deduplicated_set:
                            continue
                        deduplicated_set.add((reln.doc_name, reln.sent_num))
                        cross_sents.append(reln.sent)

        freq_dict = FreqDist([token["lemma"].lower() for tokens in cross_sents for token in tokens if token["word"].lower()
                              not in STOPWORDS and token["pos"].startswith("VB")])

        verb_set = {word for word, _ in freq_dict.most_common()[:5]}

        cross_reln = []
        deduplicated_set = set()
        for i, row in enumerate(dataset):
            if isinstance(row[u], TEntry) and isinstance(row[v], TEntry):
                for reln in row.attrs["cross_relations"]:
                    if (row[u].text.lower() in set(obj.lower() for obj in reln.subjects)
                            and row[v].text.lower() in set(obj.lower() for obj in reln.objects)):
                        if (reln.doc_name, reln.sent_num) in deduplicated_set:
                            continue
                        deduplicated_set.add((reln.doc_name, reln.sent_num))

                        accepted = False
                        for token in reln.sent[slice(*reln.relation_span)]:
                            if token["lemma"] in verb_set and token["pos"].startswith("V"):
                                accepted = True
                        if accepted:
                            cross_reln.append((reln, reln.relation))

        cross_relations[u, v] = cross_reln
    dataset.attrs[name] = cross_relations


def recalculate_graph_categories(dataset, name="column_category_scores"):
    num_rows, num_cols = len(dataset), len(dataset[0])
    dataset.attrs[name] = {}

    for j in xrange(num_cols):
        mark_cnt = 0
        for i in xrange(num_rows):
            item = dataset[i][j]
            if "best_candidate_doc" in item.attrs:
                mark_cnt += 1
        if mark_cnt < 0.2 * num_rows:
            continue

        counter = Counter()

        for i in xrange(num_rows):
            item = dataset[i][j]
            if "best_candidate_doc" in item.attrs:
                categories = read_doc_categories(title=item["best_candidate_doc"])
                for category in categories:
                    freq = get_category_frequency(category)
                    if freq > 0:
                        counter[category] += np.log2(NUM_WIKI_DOCS / freq)

        dataset.attrs[name][j] = counter


def mark_literal_ner(dataset, name="ner_annotation"):
    num_rows, num_cols = len(dataset), len(dataset[0])

    ner_annotations = {}
    for j in xrange(num_cols):
        votes = Counter()

        for i in xrange(num_rows):
            if not isinstance(dataset[i][j], TEntry):
                continue

            text = dataset[i][j].text
            annotated_obj = annotate(text, properties=NLP_PROPERTIES_NER)

            for sentence in annotated_obj["sentences"]:
                for token in sentence["tokens"]:
                    if token["word"] == text:
                        votes[token["ner"]] += 1

        if len(votes) > 0 and votes.most_common()[0][0] != 'O':
            ner_annotations[j] = votes.most_common()[0][0]

    dataset.attrs[name] = ner_annotations


def do_query(queued_questions, doc_names=None, doc_scores=None, top_n=1, n_docs=3):
    if doc_names is None:
        doc_names = []
    if doc_scores is None:
        doc_scores = []

    query_obj = {
        "questions": queued_questions,
        "doc_names": doc_names,
        "doc_scores": doc_scores,
        "return_context": False,
        "n_docs": n_docs,
        "top_n": top_n
    }

    # print(ujson.dumps(query_obj))
    return requests.post("http://localhost:5555", data=ujson.dumps(query_obj).encode()).json()


def analyze_result(target_col, result_obj, queue_state_rows, dataset, state_macro_f1s):
    for i, (robj, (sid, row)) in enumerate(zip(result_obj, queue_state_rows)):
        if len(robj) == 0:
            state_macro_f1s[sid].append(0.0)
            continue

        entry_tokens = [token["lemma"] for
                        token in
                        annotate(dataset[row][target_col].text, properties=NLP_PROPERTIES_LEMMA)
                        ["sentences"][0]["tokens"]]
        span_tokens = [token["lemma"] for sent in
                       annotate(robj[0]["span"], properties=NLP_PROPERTIES_LEMMA)
                       ["sentences"] for token in sent["tokens"]]
        entry_set, span_set = set(entry_tokens), set(span_tokens)
        union_set = entry_set | span_set

        sorted_entries = sorted(union_set)
        true_labels = [1 if key in span_tokens else 0 for key in sorted_entries]
        pred_labels = [1 if key in entry_tokens else 0 for key in sorted_entries]

        state_macro_f1s[sid].append(f1_score(true_labels, pred_labels, average="macro"))


def eval_state_results(target_col, states, dataset, lock_docs=False):
    num_rows, num_cols = len(dataset), len(dataset[0])
    completed_rows = []

    for i in xrange(num_rows):
        count = sum(1 for j in xrange(num_cols) if isinstance(dataset[i][j], TEntry))
        if count == num_cols:
            completed_rows.append(i)

    column_phrases = dataset.attrs["column_phrases"]
    queued_questions, queue_state_rows = [], []
    state_macro_f1s = defaultdict(list)

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

    doc_names, doc_scores = None, None

    if len(completed_rows) == 0:
        return []

    for sid, state in enumerate(states):
        for i in np.random.choice(completed_rows, size=min(25, len(completed_rows)), replace=False):
            question = ["Which", "is"]
            for t, (u, v) in enumerate(state.components):
                if t == 0:
                    question.append(column_phrases[u][v])
                    question.append("for")
                if t > 1:
                    question.append("and")
                if t > 0:
                    question.append(dataset[i][u].text)

            question = " ".join(question) + "?"
            queued_questions.append(question)
            queue_state_rows.append((sid, i))

            if len(queued_questions) == 512:
                if lock_docs:
                    doc_names, doc_scores = [], []
                    for _, row in queue_state_rows:
                        doc_names.append(doc_names_by_row[row][:])
                        if ("best_candidate_doc" in dataset[row][target_col].attrs and
                                dataset[row][target_col]["best_candidate_doc"] in doc_names[-1]):
                            doc_names[-1].remove(dataset[row][target_col]["best_candidate_doc"])
                        doc_scores.append([1.0 for _ in xrange(len(doc_names[-1]))])

                result_obj = do_query(queued_questions, doc_names, doc_scores)
                analyze_result(target_col, result_obj, queue_state_rows, dataset, state_macro_f1s)
                queued_questions.clear()
                queue_state_rows.clear()

    if len(queue_state_rows) > 0:
        if lock_docs:
            doc_names, doc_scores = [], []
            for _, row in queue_state_rows:
                doc_names.append(doc_names_by_row[row][:])
                if ("best_candidate_doc" in dataset[row][target_col].attrs and
                            dataset[row][target_col]["best_candidate_doc"] in doc_names[-1]):
                    doc_names[-1].remove(dataset[row][target_col]["best_candidate_doc"])
                doc_scores.append([1.0 for _ in xrange(len(doc_names[-1]))])

        result_obj = do_query(queued_questions, doc_names, doc_scores)
        analyze_result(target_col, result_obj, queue_state_rows, dataset, state_macro_f1s)

    return [np.mean(state_macro_f1s.get(sid, [0.0])) for sid in xrange(len(states))]


def search_template_for_columns(dataset, target_col, total_steps, external, ks):
    num_cols = len(dataset[0])
    num_tokens = num_cols
    column_phrases = dataset.attrs["column_phrases"]
    num_variant = [len(column_phrases[k]) for k in xrange(num_cols)]

    search_obj = BeamSearch(num_tokens, num_variant)
    search_obj.candidate_states = []

    initial_states = []
    for p in xrange(1, len(column_phrases[target_col])):
        initial_states.append(State([(target_col, p)], {target_col}, 0.0, external=external))
    scores = eval_state_results(target_col, initial_states, dataset, lock_docs=not external)
    for state, score in zip(initial_states, scores):
        state.score = score
    search_obj.candidate_states = initial_states

    for step in xrange(total_steps):
        new_candidates = search_obj.generate_next_candidates()
        new_scores = eval_state_results(target_col, new_candidates, dataset, lock_docs=not external)
        for state, score in zip(new_candidates, new_scores):
            state.score = score

        search_obj.set_k(ks[step])
        search_obj.update_candidate_scores(new_candidates)

    return sorted((state for state in search_obj.candidate_states
                                if len(state.components) > 1), reverse=True)[:10]


def search_question_template(dataset, name="candidate_templates", total_steps=2,
                             external=False, ks=None):
    num_rows, num_cols = len(dataset), len(dataset[0])

    column_phrases = {}
    for j in xrange(num_cols):
        current = []
        if j not in dataset.attrs["column_category_scores"]:
            column_phrases[j] = current
            continue

        for cate_name, cate_score in dataset.attrs["column_category_scores"][j].most_common()[:3]:
            current.extend([remove_parenthesis_on_tokens_text([token["word"]
                            for token in annotate(text)["sentences"][0]["tokens"]])
                            for text in extract_noun_phrases(cate_name.replace("_", " "))
                            ])

        current = list(set(current))
        random.shuffle(current)
        current = current[:15]
        column_phrases[j] = current

    for j in xrange(num_cols):
        if j not in dataset.attrs["ner_annotation"]:
            continue

        ner_annotation = dataset.attrs["ner_annotation"][j]
        current = NER_MAP.get(ner_annotation, [ner_annotation.lower()])
        column_phrases[j] += current
        column_phrases[j] = list(set(column_phrases[j]))

    dataset.attrs["column_phrases"] = column_phrases

    if ks is None:
        ks = [20 for _ in xrange(total_steps)]
    elif isinstance(ks, int):
        ks = [ks for _ in xrange(total_steps)]

    candidate_templates = {}
    for j in xrange(num_cols):
        candidate_templates[j] = search_template_for_columns(dataset, j, total_steps,
                                                             external, ks)

    dataset.attrs[name] = candidate_templates
