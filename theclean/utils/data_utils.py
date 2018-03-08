#!/usr/bin/env python
from .data_elements import Row, EEntry, TEntry, Dataset


def read_dataset(header, rows):
    row_objs = []
    for row in rows:
        entry_objs = []
        for entry in row:
            if entry == "<E>" or entry.strip() == "":
                entry_objs.append(EEntry(None))
            else:
                entry_objs.append(TEntry(entry, None))
        row_objs.append(Row(entry_objs))

    return Dataset(header, row_objs)


def chunk_iter(it, chunk_size):
    cnt = 0
    chunks = []
    for element in it:
        cnt += 1
        chunks.append(element)

        if cnt == chunk_size:
            yield chunks
            cnt = 0
            chunks = []
    if cnt > 0:
        yield chunks
