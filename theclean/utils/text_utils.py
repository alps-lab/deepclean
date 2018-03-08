from six.moves import xrange

import ujson
import networkx as nx
from nltk.tree import Tree

from pycorenlp import StanfordCoreNLP


NLP_SERVER = "http://localhost:9000"
NLP_PROPERTIES={
  'annotators': 'tokenize,ssplit',
  'outputFormat': 'json'
}

NLP_PROPERTIES_LEMMA = {
'annotators': "tokenize,ssplit,pos,lemma",
'outputFormat': 'json',
}

NLP_PROPERTIES_COREF = {
'annotators': "tokenize,ssplit,pos,lemma,ner,depparse,coref",
'outputFormat': 'json',
}

NLP_PROPERTIES_IE = {
'annotators': "tokenize,ssplit,pos,lemma,depparse,ner,mention,coref,natlog,openie",
'coref.algorithm': "statistical",
'outputFormat': 'json',
}

NLP_PROPERTIES_PARSE = {
"annotators": "tokenize,ssplit,parse,lemma,ner",
"outputFormat": "json",
}

NLP_PROPERTIES_NER = {
"annotators": "tokenize,ssplit,pos,lemma,depparse,ner",
"outputFormat": "json",
}

STOPWORDS = ujson.loads(
    '["!!","?!","??","!?","`","``","\'\'","-lrb-","-rrb-","-lsb-","-rsb-",",",".",":",";","\\"","\'","?","<",">","{","}","[","]","+","-","(",")","&","%","$","@","!","^","#","*","..","...","\'ll","\'s","\'m","a","about","above","after","again","against","all","am","an","and","any","are","aren\'t","as","at","be","because","been","before","being","below","between","both","but","by","can","can\'t","cannot","could","couldn\'t","did","didn\'t","do","does","doesn\'t","doing","don\'t","down","during","each","few","for","from","further","had","hadn\'t","has","hasn\'t","have","haven\'t","having","he","he\'d","he\'ll","he\'s","her","here","here\'s","hers","herself","him","himself","his","how","how\'s","i","i\'d","i\'ll","i\'m","i\'ve","if","in","into","is","isn\'t","it","it\'s","its","itself","let\'s","me","more","most","mustn\'t","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan\'t","she","she\'d","she\'ll","she\'s","should","shouldn\'t","so","some","such","than","that","that\'s","the","their","theirs","them","themselves","then","there","there\'s","these","they","they\'d","they\'ll","they\'re","they\'ve","this","those","through","to","too","under","until","up","very","was","wasn\'t","we","we\'d","we\'ll","we\'re","we\'ve","were","weren\'t","what","what\'s","when","when\'s","where","where\'s","which","while","who","who\'s","whom","why","why\'s","with","won\'t","would","wouldn\'t","you","you\'d","you\'ll","you\'re","you\'ve","your","yours","yourself","yourselves","###","return","arent","cant","couldnt","didnt","doesnt","dont","hadnt","hasnt","havent","hes","heres","hows","im","isnt","its","lets","mustnt","shant","shes","shouldnt","thats","theres","theyll","theyre","theyve","wasnt","were","werent","whats","whens","wheres","whos","whys","wont","wouldnt","youd","youll","youre","youve"]')
STOPWORDS = set(STOPWORDS)


def annotate(text, url=None, properties=None):
    if url is None:
        url = NLP_SERVER
    if properties is None:
        properties = NLP_PROPERTIES
    nlp = StanfordCoreNLP(url)
    return nlp.annotate(text, properties)


def extract_sentences(jobj, lower=False):
    # print(jobj)
    sentences = []
    for sentence in jobj["sentences"]:
        cur = [token["word"] if not lower else token["word"].lower()
               for token in sentence["tokens"]]
        sentences.append(cur)
    return sentences


def mark_tokens(src_tokens, target_tokens, lower=False):
    """
    Mark the beginning and end of target_tokens in the given src_tokens as a sublist
    :param src_tokens:
    :param target_tokens:
    :return: a list of all the intervals of src_tokens that include target_tokens, right
    exclusive.
    """
    l1, l2 = len(src_tokens), len(target_tokens)
    pos = []
    for i in xrange(l1 - l2):
        c = 0
        for j in xrange(l2):
            st, tt = src_tokens[i + j], target_tokens[j]
            if lower:
                st, tt = st.lower(), tt.lower()
            if st == tt:
                c += 1
            else:
                break
        if c == l2:
            pos.append((i, i + l2))

    return pos


def traverse_tree(tree, to_append):
    for subtree in tree:
        if isinstance(subtree, Tree):
            if subtree.label() not in ["ADJP", "ADVP", "PP"]:
                traverse_tree(subtree, to_append)

    if tree.label() == "NP":
        to_append.append(" ".join(tree.leaves()))


def extract_noun_phrases(text):
    jobj = annotate(text, properties=NLP_PROPERTIES_PARSE)["sentences"][0]
    tokens = []
    for token in jobj["tokens"]:
        if token["pos"].startswith("NNP"):
            tokens.append(token["word"])
        else:
            tokens.append(token["word"].lower())

    jobj = annotate(" ".join(tokens), properties=NLP_PROPERTIES_PARSE)["sentences"][0]

    tree = Tree.fromstring(jobj["parse"])
    noun_phrases = []
    traverse_tree(tree, noun_phrases)

    return noun_phrases



