Effective Dimensionality Reduction for Word Embeddings.

Code for the [arXiv paper](https://arxiv.org/abs/1708.03629). Accepted at NIPS 2017 LLD Workshop.

> Abstract: Word embeddings have become the basic building blocks for several natural language processing and information retrieval tasks. Pre-trained word embeddings are used in several downstream applications as well as for constructing representations for sentences, paragraphs and documents. Recently, there has been an emphasis on further improving the pre-trained word vectors through post-processing algorithms. One such area of improvement is the dimensionality reduction of the word embeddings. Reducing the size of word embeddings through dimensionality reduction can improve their utility in memory constrained devices, benefiting several real-world applications. In this work, we present a novel algorithm that effectively combines PCA based dimensionality reduction with a recently proposed post-processing algorithm, to construct word embeddings of lower dimensions. Empirical evaluations on 12 standard word similarity benchmarks show that our algorithm reduces the embedding dimensionality by 50%, while achieving similar or (more often) better performance than the higher dimension embeddings.

The word-vector evaluation code is directly used from https://github.com/mfaruqui/eval-word-vectors.  

Run the script ```algo.py``` (embedding file location is hardcoded as of now) to reproduce the algorithm and its evaluation. 

Similarly, other baselines': PCA ```pca_simple.py```, PPA+PCA ```ppa_pca.py``` and PCA+PPA ```pca_ppa.py``` results can be reproduced.

To run the algo and the baselines (as in the paper) get the embedding files ([Glove](https://nlp.stanford.edu/projects/glove/), [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)) and put the file locations as required in the code.

The code will generate and evaluate (on 12 word-similarity datasets) a modified word embedding file that is half-the-size of the original embeddings.

The algorithm can be used to generate embeddings of any size, not necessarily half.
