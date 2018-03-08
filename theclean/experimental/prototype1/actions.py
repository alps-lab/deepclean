#!/usr/bin/env python
from six.moves import cPickle

from itertools import combinations

import requests

from theclean.utils.data_elements import BinaryRelationF
from theclean.utils.data_utils import Row, TEntry, Dataset
from theclean.utils.drqa_utils import get_ranker
from theclean.utils.text_utils import annotate, extract_sentences, mark_tokens, NLP_PROPERTIES_ADVANCED
from theclean.utils.wiki_utils import read_doc, read_doc_categories


def mark_top_docs_scores(row, name="doc_scores", k=1,
                         rank_server=None):
    if rank_server is None:
        ranker = get_ranker()
        for entry in row:
            if isinstance(entry, TEntry):
                text = entry.text
                doc_names, doc_scores = ranker.closest_docs(text, k)
                doc_scores = doc_scores.tolist()
                entry[name] = (doc_names, doc_scores)
    else:
        for entry in row:
            if isinstance(entry, TEntry):
                text = entry.text
                query_obj = dict(query=text, k=k)
                req = requests.post(rank_server, json=query_obj)
                robj = req.json()
                doc_names, doc_scores = robj["doc_names"], robj["doc_scores"]
                doc_scores = [float(score) for score in doc_scores]

                entry[name] = (doc_names, doc_scores)


def mark_entry_collocations(row, name="collocations", limit=60):
    docs = {}
    entry_cols, entry_texts, entry_tokens = [], [], []
    for i, entry in enumerate(row):
        if isinstance(entry, TEntry):
            if len(entry.text.strip()) == 0:
                continue

            entry_texts.append(entry.text)
            entry_cols.append(i)

            for doc_name in entry.attributes["doc_scores"][0]:
                docs[doc_name] = read_doc(title=doc_name)[0][1]

    for text in entry_texts:
        entry_tokens.append(extract_sentences(annotate(text.lower()))[0])

    doc_paragraphs = {}

    for doc_name, text in docs.items():
        doc_paragraphs[doc_name] = []
        paragraphs = text.split("\n\n")[1:]

        for paragraph in paragraphs:
            jobj = annotate(paragraph)
            doc_paragraphs[doc_name].append((paragraph, jobj))

    collocations = []
    for doc_name, text in docs.items():
        for i, j in combinations(range(len(entry_tokens)), r=2):
            found = False

            for paragraph, jobj in doc_paragraphs[doc_name]:
                if found:
                    break

                jobj = annotate(paragraph)
                sentences = extract_sentences(jobj, lower=True)
                doc_tokens = [token for sentence in sentences
                              for token in sentence]

                marks = []

                for tokens in entry_tokens:
                    marks.append(mark_tokens(doc_tokens, tokens))

                bestl, bestb, beste = limit + 1, -1, -1
                for b1, e1 in marks[i]:
                    for b2, e2 in marks[j]:
                        if 0 <= e2 - b1 < bestl :
                            bestb, beste = b1, e2
                            bestl = e2 - b1
                        elif 0 <= e1 - b2 < bestl:
                            bestb, beste = b2, e1
                            bestl = e1 - b2

                if bestl <= limit:
                    sbeg, send, csum = 0, 0, 0
                    for sent in jobj["sentences"]:
                        csum += len(sent["tokens"])
                        if csum > bestb:
                            break
                        sbeg += 1

                    csum = 0
                    for sent in jobj["sentences"]:
                        csum += len(sent["tokens"])
                        if csum >= beste:
                            break
                        send += 1

                    # print(sbeg, send)
                    obeg, oend = (jobj["sentences"][sbeg]["tokens"][0]['characterOffsetBegin'],
                                  jobj["sentences"][send]["tokens"][-1]['characterOffsetEnd'])
                    pair = ((entry_cols[i], entry_cols[j]) if
                        entry_cols[i] < entry_cols[j] else (entry_cols[j], entry_cols[i]))
                    collocations.append(dict(pair=pair,
                                             text=paragraph[obeg:oend],
                                             doc_name=doc_name))
                    found = True

    row.attrs[name] = collocations


def _get_relations(jobj, cumlen, entry_texts, doc_name, paraid):
    triplets = []
    entry_texts_set = {text.lower() for text in entry_texts}
    int_corefs_classes, good_coref_classes, coref_reprs = {}, set(), {}

    for chainid, chain in enumerate(jobj["corefs"].values()):
        coref_reprs[chainid] = None
        for item in chain:
            sentid = item["sentNum"] - 1
            text = item["text"]
            offset = 0 if sentid == 0 else cumlen[sentid - 1]
            si, ei = item["startIndex"] - 1, item["endIndex"] - 1
            soff, eoff = si + offset, ei + offset

            if coref_reprs[chainid] is None and item["isRepresentativeMention"]:
                coref_reprs[chainid] = text

            int_corefs_classes[(soff, eoff)] = chainid
            if text.lower() in entry_texts_set:
                good_coref_classes.add(chainid)
                coref_reprs[chainid] = text

    # print(coref_reprs)

    for sentid, sentence in enumerate(jobj["sentences"]):
        offset = 0 if sentid == 0 else cumlen[sentid - 1]
        for ieobj in sentence["openie"]:
            subject_span = tuple(index + offset for index in ieobj["subjectSpan"])
            object_span = tuple(index + offset for index in ieobj["objectSpan"])

            _subject = ieobj["subject"]
            _object = ieobj["object"]

            subject_class, object_class = None, None
            if subject_span in int_corefs_classes:
                subject_class = int_corefs_classes[subject_span]
            if object_span in int_corefs_classes:
                object_class = int_corefs_classes[object_span]

            if ((subject_class is not None and subject_class in good_coref_classes)
            or (object_class is not None and object_class in good_coref_classes) or
                    (_subject.lower() in entry_texts_set) or (_object.lower() in
                                                          entry_texts_set)):
                if subject_class is None:
                    subject_class = -1
                if object_class is None:
                    object_class = -1

                triplets.append(BinaryRelationF(
                    coref_reprs.get(subject_class, None),
                    coref_reprs.get(object_class, None)
                    , ieobj["relation"],
                    _subject,
                    _object,
                   doc_name))

    return triplets


def _merge_relations(jobj, cumlen, entry_texts, relations):
    return relations


def mark_relations_and_corefs(row, name="rel_triplets"):
    docs = {}
    entry_cols, entry_texts, entry_tokens = [], [], []
    for i, entry in enumerate(row):
        if isinstance(entry, TEntry):
            if len(entry.text.strip()) == 0:
                continue

            entry_texts.append(entry.text)
            entry_cols.append(i)

            for doc_name in entry.attributes["doc_scores"][0]:
                docs[doc_name] = read_doc(title=doc_name)[0][1]

    for text in entry_texts:
        entry_tokens.append(extract_sentences(annotate(text.lower()))[0])

    triplets = []
    for doc_name, text in docs.items():
        paragraphs = text.split("\n\n")[1:]

        for paraid, paragraph in enumerate(paragraphs):
            cumlen = [0]
            jobj = annotate(paragraph, properties=NLP_PROPERTIES_ADVANCED)
            sentences = extract_sentences(jobj, lower=True)

            for sent in sentences:
                cumlen.append(cumlen[-1] + len(sent))
            cumlen = cumlen[1:]

            cur_triplets = _get_relations(jobj, cumlen, entry_texts, doc_name, paraid)
            triplets.extend(_merge_relations(jobj, cumlen, entry_texts, cur_triplets))

    row.attrs[name] = triplets


def mark_doc_categories(row, name="doc_categories"):
    doc_categories = {}
    for entry in row:
        if isinstance(entry, TEntry):
            entry_categories = []
            for doc_names, doc_scores in zip(*entry["doc_scores"]):
                for doc_name in doc_names:
                    if doc_name not in doc_categories:
                        doc_categories[doc_name] = read_doc_categories(title=doc_name)

                    entry_categories.append(doc_categories[doc_name])
            entry[name] = entry_categories


def analyze_relations_and_corefs():
    pass
