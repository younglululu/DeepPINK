# DeepPINK
DeepPINK: reproducible feature selection in deep neural networks (https://papers.nips.cc/paper/8085-deeppink-reproducible-feature-selection-in-deep-neural-networks.pdf)

(1) Software to generate knockoffs.  The input should be a n x p matrix (n is the size of samples and p is the dimension of features), better standardized. And the output is a n x 2p matrix, the first p dimension for original features and next p dimension for knockoff features. 
https://drive.google.com/file/d/1THhrMioEfZNU5-ExInXmKUNDNa3fQEcd/view?usp=sharing

Note that there are multiple ways to generate knockoffs, such as using deep neural networks (https://web.stanford.edu/group/candes/deep-knockoffs/).

(2) a sample code of DeepPINK. The input is n x 2p matrix and n x 1 labels/responses. The output contains n x 1 feature importance, n x 1 feature knockoff statistics, and the set of features selected subjected to specified FDR threshold.
