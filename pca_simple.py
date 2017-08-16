import numpy as np
import cPickle as pickle
from sklearn.decomposition import PCA
import subprocess

Glove = {}
f = open('/home/vikas/Desktop/glove.6B/glove.6B.300d.txt')
print("Loading Glove vectors.")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    Glove[word] = coefs
f.close()

print("Done.")
X_train = []
X_train_names = []
for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
pca_embeddings = {}
embedding_file = open('pca_embedding_30.txt', 'w')

# PCA with 150 dimensions.
pca =  PCA(n_components = 150)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

for i, x in enumerate(X_train_names):
        pca_embeddings[x] = X_fit[i]
        embedding_file.write("%s\t" % x)
        for t in pca_embeddings[x]:
                embedding_file.write("%f\t" % t)        
        embedding_file.write("\n")


print("Results for the Embedding")
print subprocess.check_output(["python", "all_wordsim.py", "pca_embedding_30.txt", "data/word-sim/"])
print("Results for Glove")
print subprocess.check_output(["python", "all_wordsim.py", "../glove.6B/glove.6B.300d.txt", "data/word-sim/"])
