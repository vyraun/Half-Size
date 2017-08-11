import numpy as np
import cPickle as pickle
from sklearn.decomposition import PCA
import subprocess

Glove = {}
f = open('/home/vikasraunak1994/Desktop/glove.6B/glove.6B.300d.txt')
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
#X_test = X_train[55:175]
X_train = np.asarray(X_train)
#X_test = np.asarray(X_test)

pca_embeddings = {}
embedding_file = open('pca_embedding_30.txt', 'w')

# PCA with 150 dimensions with U2[0:6] on 30 dimensions beats 300D in 9/12 tasks and some with huge margins.

# First PCA
pca =  PCA(n_components = 150)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

### Extra: Subtract Mean Too. 
# X_fit = X_fit - np.mean(X_fit)
### Does good. Not sure be

for i, x in enumerate(X_train_names):
        pca_embeddings[x] = X_fit[i]
        embedding_file.write("%s\t" % x)
	#for u in U2[0:2]:        
    #    	pca_embeddings[x] = pca_embeddings[x] - np.dot(u.transpose(),pca_embeddings[x]) * u 

        for t in pca_embeddings[x]:
                embedding_file.write("%f\t" % t)
        
        embedding_file.write("\n")
#encoded_imgs = encoder.predict(X_train)
#decoded_imgs = decoder.predict(encoded_imgs)               
#pickle.dump(Autoencoder_embeddings, open("Autoencoder_embeddings.p", "wb" ))


print("Results for the Embedding")
print subprocess.check_output(["python", "all_wordsim.py", "pca_embedding_30.txt", "data/word-sim/"])
print("Results for the 100D Glove")
print subprocess.check_output(["python", "all_wordsim.py", "../glove.6B/glove.6B.300d.txt", "data/word-sim/"])
