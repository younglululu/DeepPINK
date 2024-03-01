# DeepPINK

Yang Lu, Yingying Fan, Jinchi Lv, William Stafford Noble.  ["DeepPINK: reproducible feature selection in deep neural networks"] (https://papers.nips.cc/paper/8085-deeppink-reproducible-feature-selection-in-deep-neural-networks)
_Advances in Neural Information Processing Systems 31_ (NeurIPS), 2018.

This repository contains a Python implementation of DeepPINK.
The input is an n x 2p matrix and n x 1 labels. The output contains n x 1 feature importance values, n x 1 feature knockoff statistics, and the set of features selected subjected to the specified FDR threshold.

To use DeepPINK, you must first generate knockoffs.
Note that there are multiple ways to generate such knockoffs, such as using [deep neural networks](https://web.stanford.edu/group/candes/deep-knockoffs).

All datasets used in the DeepPINK paper are available at (https://noble.gs.washington.edu/proj/DeepPINK).
