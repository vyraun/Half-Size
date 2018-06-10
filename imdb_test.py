import pickle
import numpy as np
from keras.datasets import imdb
from keras.utils.data_utils import get_file

#top_words = 5000
#test_split = 0.30
#(X, y), (X_test, y_test) = imdb.load_data()  # num_words=top_words

path = get_file('imdb_full.pkl',
                 origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                 md5_hash='d091312047c43cf9e4e38fef92437263')
f = open(path, 'rb')
(X, y), (test_data, test_labels) = pickle.load(f)
        
print("Reading Done")
        
X, y = np.array(X), np.array(y)
print ("Total Training Data Points %s" % len(y))
#print ("X Shape = {}, y Shape = {}".format(X.shape, y.shape))
#print ("Sample X = {0}".format(X[0]))
#print ("Sample y = {0}".format(y[0]))

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
print("An Input Vector Sample = {}".format(X[0]))
print("A Transformed Label Sample = {}".format(y[0]))

rX = X.copy()
rvec = TfidfEmbeddingVectorizer(rw2v)
rvec.fit(rX, y)
rX = rvec.transform(rX)
print("The Reduced Embedding Matrix Shape:")
print(rX.shape)

wX = X.copy()
wvec = TfidfEmbeddingVectorizer(w2v)
wvec.fit(wX, y)
wX = wvec.transform(wX)
print("The Non-Reduced Embedding Matrix Shape:")
print(wX.shape)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

print("A Transformed (Reduced) Input Vector Sample = {}".format(rX[0]))
print("A Transformed Label Sample = {}".format(y[0]))
print("The Label Classes = {}".format(le.classes_))

print("Starting the Model Training for Reduced Data")
rclf = LinearSVC(random_state=0)
rclf.fit(rX, y)
print("Training set score: %f" % rclf.score(rX, y))

print("Starting the Model Training for Non-Reduced Data")
wclf = LinearSVC(random_state=0)
wclf.fit(wX, y)
print("Training set score: %f" % wclf.score(wX, y))

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
