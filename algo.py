import numpy as np
import pickle as pickle
from sklearn.decomposition import PCA
import subprocess

ORIGINAL_DIMS = 50
ORIGINAL_FILE_NAME = './GloVe/glove.6B.50d.txt'
FINAL_DIMS = 25
FINAL_FILE_NAME = "./GloVe/pca_embed2.txt"

Glove = {}
f = open(ORIGINAL_FILE_NAME, 'r', encoding="utf-8")

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

# PCA to get Top Components
pca =  PCA(n_components = ORIGINAL_DIMS)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

z = []

# Removing Projections on Top Components
for i, x in enumerate(X_train):
	for u in U1[0:7]:        
        	x = x - np.dot(u.transpose(),x) * u 
	z.append(x)

z = np.asarray(z)

# PCA Dim Reduction
pca =  PCA(n_components = FINAL_DIMS)
X_train = z - np.mean(z)
X_new_final = pca.fit_transform(X_train)


# PCA to do Post-Processing Again
pca =  PCA(n_components = FINAL_DIMS)
X_new = X_new_final - np.mean(X_new_final)
X_new = pca.fit_transform(X_new)
Ufit = pca.components_

X_new_final = X_new_final - np.mean(X_new_final)

final_pca_embeddings = {}
embedding_file = open(FINAL_FILE_NAME, 'w', encoding="utf-8")

for i, x in enumerate(X_train_names):
        final_pca_embeddings[x] = X_new_final[i]
        embedding_file.write("%s " % x)
        for u in Ufit[0:7]:
            final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(),final_pca_embeddings[x]) * u 

        for t in final_pca_embeddings[x]:
                embedding_file.write("%f " % t)
        
        embedding_file.write("\n")


print("Results for the Embedding")
print((subprocess.check_output(["python", "all_wordsim.py", FINAL_FILE_NAME, "data/word-sim/"])).decode("utf-8"))
print("Results for Glove")
print((subprocess.check_output(["python", "all_wordsim.py", ORIGINAL_FILE_NAME, "data/word-sim/"])).decode("utf-8"))