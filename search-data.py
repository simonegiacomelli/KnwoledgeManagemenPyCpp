from pprint import pprint  # pretty-printer
from collections import defaultdict
from re import finditer
from gensim import corpora
from gensim import models
import gensim

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

# process names
documents = [d for d in [process_line(line) for line in lines]]
frequency = defaultdict(int)

for query_string in documents:
    for token in query_string[0]:
        frequency[token] += 1


# remove token that has frequency 1
def remove_infrequent(doc):
    tokens = [token for token in doc[0] if frequency[token] > 1]
    doc[0] = tokens
    return doc


# and exclude documents that does not has tokens
documents = [remove_infrequent(doc) for doc in documents]
documents = [d for d in documents if len(d[0]) > 0]

texts = [d[0] for d in documents]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

query_string = "Optimizer that implements the Adadelta algorithm".lower()
query_string = "Optimizer Adadelta".lower()
query_words = query_string.lower().split()
query_bow = dictionary.doc2bow(query_words)

from gensim import similarities


def print_results(note, sims):
    print(note)
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for i, tup in enumerate(sims):
        if i > 5:
            break
        print(tup, documents[tup[0]])


def freq_query_docs():
    index = similarities.MatrixSimilarity(corpus)
    sims = index[query_bow]  # perform a similarity query against the corpus
    print_results('FREQ', sims)


def lsi_query_docs():
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=200)
    lsi_index = similarities.MatrixSimilarity(lsi_model[corpus])
    lsi_query_vec = lsi_model[query_bow]  # convert the query to LSI space
    sims = lsi_index[lsi_query_vec]  # perform a similarity query against the corpus
    print_results('LSI', sims)


def tf_idf_query_docs():
    model = models.TfidfModel(corpus, id2word=dictionary)
    index = similarities.MatrixSimilarity(model[corpus])
    query_vec = model[query_bow]  # convert the query to LSI space
    sims = index[query_vec]  # perform a similarity query against the corpus
    print_results('TF-IDF', sims)


# todo the output can be more polished: print ranking and some description to better show what are the hits
def doc2vec_query_docs():
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def read_corpus():
        for i, line in enumerate(texts):
            print(i, line)
            yield gensim.models.doc2vec.TaggedDocument(line, [i])

    train_corpus = list(read_corpus())
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=10)

    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    vector = model.infer_vector(query_words)
    sims = model.docvecs.most_similar([vector], topn=5)
    print('doc2vec')
    for i, rank in sims:
        print(i, rank, documents[i])


doc2vec_query_docs()
freq_query_docs()
tf_idf_query_docs()
lsi_query_docs()
