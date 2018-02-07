# ALIGN, a computational tool for multi-level language analysis
## Analyzing Linguistic Interactions with Generalizable techNiques

by N. Duran, A. Paxton, & R. Fusaroli

`align` is a Python library for extracting quantitative, reproducible metrics of multi-level alignment between two speakers in naturalistic language corpora.

## Installation

`align` may downloaded directly using `pip`.

To download the stable version released on PyPI:
```
pip install align
```

To download directly from our GitHub repo:
```
pip install git+https://github.com/nickduran/align-linguistic-alignment.git
```

## Requirements

Note that `GoogleNews-vectors-negative300.bin` and `stanford-postagger-full-2017-06-09` must be downloaded separately. Once downloaded, add to `align_package` folder.  

`GoogleNews-vectors-negative300.bin` can be downloaded [HERE](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) (right click and open in new window) from https://code.google.com/archive/p/word2vec/

`stanford-postagger-full-2017-06-09` can be downloaded [HERE](https://nlp.stanford.edu/software/stanford-postagger-full-2017-06-09.zip) from https://nlp.stanford.edu/software/tagger.shtml#Download 

## Attribution

If you find the package useful, please cite our manuscript:

>Duran, N., Paxton, A., & Fusaroli, R (*submitted*). ALIGN: Analyzing
>    Linguistic Interactions with Generalizable techNiques.

## Licensing of example data

* **CHILDES**: Example corpus "Kuczaj Corpus" by Stan Kuczaj is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License (https://childes.talkbank.org/access/Eng-NA/Kuczaj.html). Kuczaj, S. (1977). The acquisition of regular and irregular past tense forms. *Journal of Verbal Learning and Verbal Behavior, 16*, 589–600.
