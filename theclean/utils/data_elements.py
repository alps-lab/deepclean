from collections import namedtuple


class Dataset(object):

    def __init__(self, header, rows, attributes=None):
        self.header = header
        self.rows = rows
        if attributes is None:
            attributes = {}
        self.attrs = attributes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


class Row(object):

    def __init__(self, entries=None, attributes=None):
        self.entries = entries
        if attributes is None:
            attributes = {}
        self.attrs = attributes

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


class Entry(object):

    def __init__(self, *args, **kwargs):
        pass


class EmptyEntry(Entry):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return "<E>"


class TextEntry(Entry):

    def __init__(self, text, **kwargs):
        super().__init__(**kwargs)
        self.text = text

    def __str__(self):
        return self.text


class AttributedEntry(Entry):

    def __init__(self, attributes=None, **kwargs):
        super().__init__(**kwargs)
        if attributes is None:
            attributes = {}
        self.attrs = attributes

    def __getitem__(self, item):
        return self.attrs[item]

    def __setitem__(self, key, value):
        if self.attrs is None:
            self.attributes = {}
        self.attrs[key] = value

    def __delitem__(self, key):
        del self.attrs[key]

    def get(self, item, default=None):
        return self.attrs.get(item, default)


class TEntry(TextEntry, AttributedEntry):

    def __init__(self, text, attributes):
        super().__init__(text=text, attributes=attributes)


class EEntry(EmptyEntry, AttributedEntry):

    def __init__(self, attributes):
        super().__init__(attributes=attributes)


BinaryRelationI = namedtuple("BinaryRelationI", ["subject", "object", "relation",
                                               "doc_name",
                                               "paragraph", "subject_span",
                                               "object_span"])

BinaryRelationF = namedtuple("BinaryRelationF", ["subject_repr", "object_repr",
                                                 "relation",
                                                "subject", "object",
                                                 "doc_name"])

#
BinaryRelationS = namedtuple("BinaryRelationS", ["subject_repr", "object_repr", "relation",
                                                 "subjects", "objects", "doc_name", "sent_num", "sent",
                                                 "object_span", "target_span", "relation_span"])

# Compact version of Binary Relation.
BinaryRelationC = namedtuple("BinaryRelationC", ["subject_repr", "object_repr", "relation",
                                                 "subjects", "objects", "doc_name",
                                                 "object_span", "target_span", "relation_span"])


SimpleRelation = namedtuple("SimpleRelation", field_names=["subject", "object", "relation"])

# indices are all zero based.
TextSpan = namedtuple("TextSpan", ["doc_name", "sent_num", "start_index", "end_index"])


def to_compact_relation(reln):
    return BinaryRelationC(reln.subject_repr, reln.object_repr,
                           reln.relation, reln.subjects,
                           reln.objects, reln.doc_name, reln.object_span,
                           reln.target_span, reln.relation_span)

