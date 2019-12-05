from pprint import pprint  # pretty-printer
from collections import defaultdict
from re import finditer

# remove common words and tokenize
stoplist = set('test tests main'.split())
with open('data.csv', 'r') as f:
    lines = f.read().splitlines(keepends=False)


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_entity(id):
    if '_' in id:
        parts = id.split('_')
    else:
        parts = camel_case_split(id)
    parts = [p.lower() for p in parts if p.strip() != '']
    return parts


def process_line(line):
    fields = line.split(',')
    name = fields[0]
    parts = split_entity(name)
    parts = [p for p in parts if p not in stoplist]
    fields.insert(0, parts)
    return fields


# todo check './tensorflow/tensorflow/core/kernels/data/flat_map_dataset_op.cc line 46
# todo it does not detect class FlatMapDatasetOp::Dataset : public DatasetBase {

# process names and exclude those that does not has accepted words
documents = [d for d in [process_line(line) for line in lines] if len(d[0]) > 0]
pprint(documents)
exit()
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]
