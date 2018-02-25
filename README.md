# ALIGN, a computational tool for multi-level language analysis

`align` is a Python library for extracting quantitative, reproducible metrics of multi-level alignment between two speakers in naturalistic language corpora. The method was introduced in "ALIGN: Analyzing Linguistic Interactions with Generalizable techNiques" (Duran, Paxton, & Fusaroli, *submitted*).

## Try out `align` with Binder

Interested in seeing how `align` works, but not sure if you want to install it yet? Try it out through Binder. Click the "launch" button to get a complete cloud environment to try out the ALIGN pipeline on our tutorials.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/nickduran/align-linguistic-alignment/master)

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

## Additional tools required for some `align` options

The Google News pre-trained word2vec vectors (`GoogleNews-vectors-negative300.bin`) and the Stanford part-of-speech tagger (`stanford-postagger-full-2017-06-09`) are required for some optional `align` parameters but must be downloaded separately.

* Google News: https://code.google.com/archive/p/word2vec/ (page) or https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing (direct download)

* Stanford POS tagger: https://nlp.stanford.edu/software/tagger.shtml#Download (page) or https://nlp.stanford.edu/software/stanford-postagger-full-2017-06-09.zip (direct download)

## Tutorials

We also provide tutorials on how to use `align` in this repository (in the `examples` directory). We are in the process of adding more tutorials and would welcome additional tutorials by interested contributors.

## Attribution

If you find the package useful, please cite our manuscript:

>Duran, N., Paxton, A., & Fusaroli, R. (*submitted*). ALIGN: Analyzing
>    Linguistic Interactions with Generalizable techNiques. http://doi.org/10.17605/OSF.IO/KX8UR

## Licensing of example data

* **CHILDES**:
    * Example corpus "Kuczaj Corpus" by Stan Kuczaj is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License (https://childes.talkbank.org/access/Eng-NA/Kuczaj.html):
    > Kuczaj, S. (1977). The acquisition of regular and irregular past tense
    >     forms. *Journal of Verbal Learning and Verbal Behavior, 16*, 589–600.
