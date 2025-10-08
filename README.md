# Multimedia Recommendation by Integrating Modality-Aware Diffusion and Similarity-Based Graph Refinement

This repository provides the official PyTorch implementation of ASTRO, as introduced in the following paper:

> Multimedia Recommendation by Integrating Modality-Aware Diffusion and Similarity-Based Graph Refinement
> Proceedings of the ACM Web Conference 2026 (WWW '26)


### Requirements
The code has been tested running under Python 3.6.13. The required packages are as follows:
- ```gensim==3.8.3```
- ```pytorch==2.4.0+cu124```
- ```sentence_transformers=2.2.0```
- ```pandas```
- ```numpy```
- ```scipy```
- ```tqdm```

### Dataset Preparation
#### Dataset Download
*Baby, Beauty, and Toys & Games*: Download 5-core reviews data, meta data, and image features from [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/links.html). Put data into the directory data/{folder}/meta-data/.

*Men Clothing and Women Clothing*: Download Amazon product dataset provided by [MAML](https://github.com/liufancs/MAML). Put data folder into the directory data/.

#### Dataset Preprocessing
Run ```python build_data.py --name={Dataset}```

### Acknowledgement
The structure of this code is largely based on [MONET](https://github.com/Kimyungi/MONET). Thank for their work.
