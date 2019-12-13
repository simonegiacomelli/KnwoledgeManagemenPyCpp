from collections import defaultdict
from re import finditer

import gensim
from gensim import corpora
from gensim import models

# remove common words and tokenize
stoplist = set('main'.split())
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
    fields = line.split(',')[1:]
    name = fields[0]
    parts = split_entity(name)
    parts = [p for p in parts if p not in stoplist]
    fields.insert(0, parts)
    return fields

# process names
all_documents_dict = {d[1]: d for d in [process_line(line) for line in lines]}

documents_no_dup = list(all_documents_dict.values())

frequency = defaultdict(int)

for doc in documents_no_dup:
    for token in doc[0]:
        frequency[token] += 1


def remove_infrequent(doc):
    # remove token that has frequency 1
    # tokens = [token for token in doc[0] if frequency[token] > 1]
    # doc[0] = tokens

    l = len([t for t in doc[0] if t == 'test' or t == 'tests'])
    if l > 0:
        doc[0] = []
    tokens = [token for token in doc[0] if len(token) > 2]
    doc[0] = tokens
    return doc


# and exclude documents that does not has tokens
documents = [remove_infrequent(doc) for doc in documents_no_dup]
documents = [d for d in documents if len(d[0]) > 0]

texts = [d[0] for d in documents]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

from gensim import similarities

# freq
freq_matrix_sim = similarities.MatrixSimilarity(corpus)

# lsi
lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=200)
lsi_martrix_sim = similarities.MatrixSimilarity(lsi_model[corpus])

# tfidf
tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
tfidf_matrix_sim = similarities.MatrixSimilarity(tfidf_model[corpus])

# doc2vec
def read_corpus():
    for i, line in enumerate(texts):
        yield gensim.models.doc2vec.TaggedDocument(line, [i])


doc2vec_train_corpus = list(read_corpus())
doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=0, epochs=60)

doc2vec_model.build_vocab(doc2vec_train_corpus)

doc2vec_model.train(doc2vec_train_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
engines = ['FREQ', 'LSI', 'TF-IDF', 'DOC2VEC']


class Query:
    def __init__(self, query):
        query_string = query.lower()
        self.query_words = query_string.lower().split()
        self.query_bow = dictionary.doc2bow(self.query_words)
        self.res_freq = []
        self.res_lsi = []
        self.res_tfidf = []
        self.res_doc2vec = []

    def results(self):
        return [self.res_freq, self.res_lsi, self.res_tfidf, self.res_doc2vec]

    def execute(self):
        self.freq_query_docs()
        self.tf_idf_query_docs()
        self.lsi_query_docs()
        self.doc2vec_query_docs()
        return self.results()

    def print(self):
        self.execute()
        self.print_results('FREQ', self.res_freq)
        self.print_results('LSI', self.res_lsi)
        self.print_results('TF-IDF', self.res_tfidf)
        self.print_results('doc2vec', self.res_doc2vec)

    def freq_query_docs(self):
        sims = freq_matrix_sim[self.query_bow]  # perform a similarity query against the corpus
        self.res_freq = self.top5(sims)

    def lsi_query_docs(self):
        lsi_query_vec = lsi_model[self.query_bow]  # convert the query to LSI space
        sims = lsi_martrix_sim[lsi_query_vec]  # perform a similarity query against the corpus
        self.res_lsi = self.top5(sims)

    def tf_idf_query_docs(self):
        query_vec = tfidf_model[self.query_bow]  # convert the query to LSI space
        sims = tfidf_matrix_sim[query_vec]  # perform a similarity query against the corpus
        self.res_tfidf = self.top5(sims)

    def doc2vec_query_docs(self):
        vector = doc2vec_model.infer_vector(self.query_words)
        self.res_doc2vec = doc2vec_model.docvecs.most_similar([vector], topn=5)

    def top5(self, sims):
        res = sorted(enumerate(sims), key=lambda item: -item[1])[:5]
        return res

    def print_results(self, note, sims):
        print()
        print(note)
        for i, tup in enumerate(sims):
            pos, sim = tup
            doc = documents[pos]
            print()
            print(f'#{i + 1} Entity name: {doc[1]}')
            print(f'   File: {doc[2]}')
            print(f'   Line: {doc[3]}')
            print(f'   Type: {doc[4]}')
