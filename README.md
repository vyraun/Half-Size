Effective Dimensionality Reduction for Word Embeddings.

Code for the [arXiv paper](https://arxiv.org/abs/1708.03629).

The word-vector evaluation code is directly used from https://github.com/mfaruqui/eval-word-vectors.  

Run the script algo.py (embedding file location is hardcoded as of now) to reproduce the algorithm and its evaluation. Similarly, other baseline (PCA: pca_simple.py), PPA+PCA (ppa_pca.py) and PCA+PPA (pca_ppa.py) results can be reproduced.

To run the algo and the baselines (as in the paper) get the embedding files ([Glove](https://nlp.stanford.edu/projects/glove/), [Fastext](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)) and put the file locations as required in the code.

The code will generate and evaluate (on 12 word-similarity datasets) a modified word embedding file that is half the size of the original embeddings.

The algorithm can be used to generate embeddings of any size, not necessaeility half.
