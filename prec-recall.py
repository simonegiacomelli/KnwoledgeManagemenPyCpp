import math

import query
import numpy as np

docmap = dict()
for idx, doc in enumerate(query.documents):
    if (doc[1] == '_is_py2_name_constant'):
        print(doc)
    docmap[(doc[1], doc[2])] = idx


class Measure:
    def __init__(self, q, results, correct_idx):
        repr = ''
        pos = 0
        for i, tup in enumerate(results):
            repr += f',{query.documents[tup[0]]}'
            if correct_idx == tup[0]:
                pos = i + 1

        if pos != 0:
            self.found = 1
            self.prec = pos / len(results)
            st = f'in position {pos}'
        else:
            self.found = 0
            self.prec = 0
            st = 'not found'

        print('  doc', st, repr)
        self.pos = pos


class GroundTruthRow:
    def __init__(self, line):
        p = line.split(',')
        self.query = query.Query(p[0])
        self.entity = p[1]
        self.file = p[2]
        self.correct_idx = docmap.get((self.entity, self.file), -1)
        if self.correct_idx == -1:
            raise Exception(f'Ground truth not found in documents. Offending line {p}')

        print(query.documents[self.correct_idx], "query: " + p[0])
        # execute returns 4 items, one for each engine
        self.results = [Measure(p[0], r, self.correct_idx) for r in self.query.execute()]

        print('precision')


def main():
    with open('ground-truth.txt', 'r') as f:
        gt_lines = [l for l in f.read().split('\n') if not l.startswith('#')]

    gts = [GroundTruthRow(line) for line in gt_lines]
    for engine in range(len(query.engines)):
        prec = np.average([g.results[engine].prec for g in gts])
        hit = np.sum([g.results[engine].found for g in gts])
        recall = hit / len(gts)
        print(query.engines[engine], f'precision: {prec} recall:{recall}')


if __name__ == '__main__':
    main()
