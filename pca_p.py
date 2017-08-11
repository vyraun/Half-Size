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

#print("X_train's shape: ", X_train.shape)
#print("Example: ", X_train[0, :10])

pca_embeddings = {}
embedding_file = open('pca_embedding_30.txt', 'w')

# PCA with 150 dimensions with U2[0:6] on 30 dimensions beats 300D in 9/12 tasks and some with huge margins.

# First PCA
pca =  PCA(n_components = 300)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

#print("X_fit's shape: ", X_fit.shape)
#print("Example: ", X_fit[0,:10])

#print("Final PCA Embeddings: ", X_fit2.shape)
#print("Final Top Components: ", U2.shape)

pca_embeddings = {}

z = []

# for i, x in enumerate(X_train):

# 	for u in U1[0:7]:        
#         x = x - np.dot(u.transpose(),x) * u 
#     z.append(x)


for i, x in enumerate(X_train):
    for u in U1[0:7]:
        x = x - np.dot(u.transpose(),x) * u
    z.append(x)
    


z = np.asarray(z).astype(np.float32)    

# First PCA
pca =  PCA(n_components = 150)
X_train = z - np.mean(z)
X_new = pca.fit_transform(X_train)

print("X_new's shape: ", X_new.shape)
print("Example: ", X_new[0, :10])

final_pca_embeddings = {}
embedding_file = open('pca_embedding_30.txt', 'w')

for i, x in enumerate(X_train_names):
        final_pca_embeddings[x] = X_new[i]
        embedding_file.write("%s\t" % x)
    #for u in Ufit:        
            #final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(),final_pca_embeddings[x]) * u 

        for t in final_pca_embeddings[x]:
                embedding_file.write("%f\t" % t)
        
        embedding_file.write("\n")


print("Results for the Embedding")
print subprocess.check_output(["python", "all_wordsim.py", "pca_embedding_30.txt", "data/word-sim/"])
print("Results for the 100D Glove")
print subprocess.check_output(["python", "all_wordsim.py", "../glove.6B/glove.6B.300d.txt", "data/word-sim/"])
