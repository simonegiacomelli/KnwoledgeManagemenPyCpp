import math

import query
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

docmap = dict()
for idx, doc in enumerate(query.documents):
    if (doc[1] == 'Adadelta'):
        print(doc)
    docmap[(doc[1])] = idx
LSI_IDX = 1


class Measure:
    def __init__(self, q, results, correct_idx):
        self.q = q
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
        self.correct_idx = docmap.get((self.entity), -1)
        if self.correct_idx == -1:
            raise Exception(f'Ground truth not found in documents. Offending line {p}')

        print(query.documents[self.correct_idx], "query: " + p[0])
        # execute returns 4 items, one for each engine
        self.results = [Measure(p[0], r, self.correct_idx) for r in self.query.execute()]

        print('precision')


with open('ground-truth.txt', 'r') as f:
    gt_lines = [l for l in f.read().split('\n') if not l.startswith('#')]

lsi = None
gts = [GroundTruthRow(line) for line in gt_lines]
for engine in range(len(query.engines)):

    results = [g.results[engine] for g in gts]

    if engine == 1:
        lsi = results

    prec = np.average([g.prec for g in results])
    hit = np.sum([g.found for g in results])
    recall = hit / len(gts)
    print()
    print(query.engines[engine] + ':')
    print(f'Average precision = {np.round(prec, 1)}')
    print(f'Recall = {np.round(recall, 1)}')


def calc_tsne_lsi():
    all_results = []
    hue = []
    size = []
    for i, gt in enumerate(gts):
        bows = [query.corpus[hit[0]] for hit in gt.query.res_lsi] + [gt.query.query_bow]
        lsi_query_vec = [[e[1] for e in query.lsi_model[b]] for b in bows]
        all_results += lsi_query_vec
        # point colors
        # col = [f'col{i}'] * len(lsi_query_vec)
        col = [gt.entity] * len(lsi_query_vec)
        hue += col
        # point marker style
        size += ['query']
        size += (['hit'] * (len(lsi_query_vec) - 1))
    return all_results, hue, size


def calc_tsne_doc2vec():
    all_results = []
    hue = []
    size = []
    for i, gt in enumerate(gts):
        print()
        print(gt.entity)
        query_vec = [query.doc2vec_model.infer_vector(gt.query.query_words)]
        for hit_idx, sim in gt.query.res_doc2vec:
            doc_words = query.documents[hit_idx][0]
            doc_vec = query.doc2vec_model.infer_vector(doc_words)
            query_vec += [doc_vec]

        all_results += query_vec

        col = [gt.entity] * len(query_vec)
        hue += col
        # point marker style
        size += ['query']
        size += (['hit'] * (len(query_vec) - 1))
    return all_results, hue, size


def plot_tsne(filename, all_results, hue, size):
    tsne = TSNE(n_components=2, verbose=0, perplexity=2, n_iter=3000)
    tsne_results = tsne.fit_transform(all_results)
    df_subset = pd.DataFrame()
    df_subset['x'] = tsne_results[:, 0]
    df_subset['y'] = tsne_results[:, 1]

    plt.figure(figsize=(9, 9))
    sns.scatterplot(
        x="x", y="y",
        hue=hue,
        size=size,
        data=df_subset,
        legend="full",
        alpha=0.6
    )
    plt.savefig(filename)
    # plt.show()


plot_tsne('plot-doc2vec.png', *calc_tsne_doc2vec())
plot_tsne('plot-lsi.png', *calc_tsne_lsi())
