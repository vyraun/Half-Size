import numpy as np

print("Reading the Training Dataset")

X, y = [], []
with open("downstream_datasets/20ng-train-no-stop.txt", "r") as infile:
    for line in infile:
        label, text = line.split("\t")
        # texts are already tokenized, just split on space
        # in a real case we would use e.g. spaCy for tokenization
        # and maybe remove stopwords etc.
        X.append(text.split())
        y.append(label)

        
print("Reading Done")
        
X, y = np.array(X), np.array(y)
print ("total training data points %s" % len(y))
print ("Sample X = {0}".format(X[0]))
print ("Sample y = {0}".format(y[0]))

print("Reading the Word Vectors")

with open("glove.6B.200d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}
    
with open("pca_embedding_30.txt", "rb") as lines:
    rw2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}
    
print("Reading Done")

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec)))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec): # embedding dictionary is passed
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec)))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

print("Transforming the Training Data")

vec = TfidfEmbeddingVectorizer(rw2v)
vec.fit(X, y)
X = vec.transform(X)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

print("A Transformed Input Vector Sample = {}".format(X[0]))
print("A Transformed Label Sample = {}".format(y[0]))

print("Starting the Model Training")
clf = LinearSVC(random_state=0)
clf.fit(X, y)

print("Training set score: %f" % clf.score(X, y))

"""
LinearSVM_rw2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(rw2v)),
    ("extra trees", LinearSVC(random_state=0))])
LinearSVM_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", LinearSVC(random_state=0))])

all_models = [
    ("LinearSVM_rw2v_tfidf", LinearSVM_rw2v_tfidf),
    ("LinearSVM_w2v_tfidf", LinearSVM_w2v_tfidf)
]


import tabulate
from tabulate import tabulate

from sklearn.cross_validation import cross_val_score
unsorted_scores = [(name, cross_val_score(model, X, y, cv=None).mean()) for name, model in all_models]
scores = sorted(unsorted_scores, key=lambda x: -x[1])


print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
"""
