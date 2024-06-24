# ALIGN, a computational tool for multi-level language analysis (optimized for Python 3.10)

`align` is a Python library for extracting quantitative, reproducible
metrics of multi-level alignment between two speakers in naturalistic
language corpora. The method was introduced in "ALIGN: Analyzing
Linguistic Interactions with Generalizable techNiques" (Duran, Paxton, &
Fusaroli, 2019; Psychological Methods).

Examples of papers relying on the ALIGN library:

* Duran, N. D., Paige, A., & D'Mello, S. K. (2024). Multi‐Level Linguistic Alignment in a Dynamic Collaborative Problem‐Solving Task. Cognitive Science, 48(1), e13398. https://doi.org/10.1111/cogs.13398
* Dideriksen, C., Christiansen, M. H., Tylén, K., Dingemanse, M., & Fusaroli, R. (2023). Quantifying the interplay of conversational devices in building mutual understanding. Journal of Experimental Psychology: General, 152(3), 864. Pre-print: https://doi.org/10.31234/osf.io/a5r74
* Dideriksen, C., Christiansen, M. H., Dingemanse, M., Højmark‐Bertelsen, M., Johansson, C., Tylén, K., & Fusaroli, R. (2023). Language‐Specific Constraints on Conversation: Evidence from Danish and Norwegian. Cognitive Science, 47(11), e13387. Pre-print: https://doi.org/10.31234/osf.io/t3s6c.
* Fusaroli, R., Weed, E., Rocca, R., Fein, D., & Naigles, L. (2023). Caregiver linguistic alignment to autistic and typically developing children: A natural language processing approach illuminates the interactive components of language development. Cognition, 236, 105422. Pre-print: https://doi.org/10.31234/osf.io/ysjec
* Fusaroli, R., Weed, E., Rocca, R., Fein, D., & Naigles, L. (2023). Repeat After Me? Both Children With and Without Autism Commonly Align Their Language With That of Their Caregivers. Cognitive Science, 47(11), e13369. DOI: 10.31234/osf.io/m8fhk.
* Tylén, K., Fusaroli, R., Østergaard, S. M., Smith, P., & Arnoldi, J. (2023). The Social Route to Abstraction: Interaction and Diversity Enhance Performance and Transfer in a Rule‐Based Categorization Task. Cognitive Science, 47(9), e13338.
* Trujillo, J. P., Dideriksen, C., Tylén, K., Christiansen, M. H., & Fusaroli, R. (2023). The dynamic interplay of kinetic and linguistic coordination in Danish and Norwegian conversation. Cognitive Science, 47(6), e13298.

<!--
## Try out `align` with Binder

Interested in seeing how `align` works, but not sure if you want to install it
yet? Try it out through Binder. Click the "launch" button to get a complete
cloud environment to try out the ALIGN pipeline on our Python tutorials (the CHILDES
  tutorial is currently the only one fully operational). The process for Binder to launch may
  take several minutes.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/nickduran/align-linguistic-alignment/master)
-->

## Installation

`align` may be downloaded directly using `pip`.

To download the stable version released on PyPI:

```
pip install align
```

Or to update:

```
pip install align --upgrade
```

> And it's always good practice to install a package like `align`, which has several dependencies (see `requirements.txt`), in a virtual environment.

> **Anaconda users:** The above should work in the vast majority of cases. However, if you prefer an easy way to install `align` within a virtual environment in one go, or you are experiencing problems with trying to update `align`, a YAML file has been provided to streamline things. Just follow these simple steps:

> 1. Download the `environment.yml` file and navigate to the folder where it has been downloaded
> 2. Run the following command in Terminal: `conda env create -f environment.yml`
> 3. Be sure to activate the new enviroment (i.e., `conda activate align0.1.1`) before running any `align` analyses (such as the tutorials; see below)

If you experience any problems, please put them in the "Issues" section of this repository.

<!--
To download directly from our GitHub repo:

```
pip install git+https://github.com/nickduran/align-linguistic-alignment.git
```
-->

## Quick documentation

ALIGN consists of two primary modules for conducting analyses, `prepare_transcripts` and `calculate_alignment`. To get a quick glance of the functions contained within each module, please check out the following:

- `prepare_transcripts`: https://nickduran.github.io/align-linguistic-alignment/prepare_transcripts.html

- `calculate_alignment`: https://nickduran.github.io/align-linguistic-alignment/calculate_alignment.html

## Additional tools required for some `align` options

The Google News pre-trained word2vec vectors (`GoogleNews-vectors-negative300.bin`)
and the Stanford part-of-speech tagger (`stanford-postagger-full-2020-11-17`)
are required for some optional `align` parameters but must be downloaded
separately. Please see the tutorials for more information.

- Google News: https://code.google.com/archive/p/word2vec/ (page) or
  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
  (direct download)

- Stanford POS tagger: https://nlp.stanford.edu/software/tagger.shtml#Download (page)
  or https://nlp.stanford.edu/software/stanford-tagger-4.2.0.zip
  (direct download)

## Tutorials

We created Jupyter Notebook tutorials to provide an easily accessible
step-by-step walkthrough on how to use `align`. Below are descriptions of the
current tutorials that can be found in the `examples` directory within this
repository. If unfamiliar with Jupyter Notebooks, instructions for installing
and running can be found here: http://jupyter.org/install. We recommend installing
Jupyter using Anaconda. Anaconda is a widely-used Python data science platform
that helps streamline workflows.

- **Jupyter Notebook 1: CHILDES**

  - This tutorial walks users through an analysis of conversations from a
    single English corpus from the CHILDES database (MacWhinney,
    2000)---specifically, Kuczaj’s Abe corpus (Kuczaj, 1976). We analyze the
    last 20 conversations in the corpus in order to explore how ALIGN can be
    used to track multi-level linguistic alignment between a parent and child
    over time, which may be of interest to developmental language researchers.
    Specifically, we explore how alignment between a parent and a child
    changes over a brief span of developmental trajectory.

- **Jupyter Notebook 2: Devil's Advocate**
  - This tutorial walks users throught the analysis reported in (Duran,
    Paxton, & Fusaroli, 2019). The corpus consists of 94 written
    transcripts of conversations, lasting eight minutes each, collected from
    an experimental study of truthful and deceptive communication. The goal
    of the study was to examine interpersonal linguistic alignment between
    dyads across two conversations where participants either agreed or
    disagreed with each other (as a randomly assigned between-dyads condition)
    and where one of the conversations involved the truth and the other
    deception (as a within-subjects condition).

We are in the process of adding more tutorials and would welcome additional
tutorials by interested contributors.

## Attribution

If you find the package useful, please cite our manuscript:

> Duran, N., Paxton, A., & Fusaroli, R. (2019). ALIGN: Analyzing
> Linguistic Interactions with Generalizable techNiques. _Psychological Methods_. http://dynamicog.org/papers/

## Licensing of example data

- **CHILDES**

  - Example corpus "Kuczaj Corpus" by Stan Kuczaj is licensed under a
    Creative Commons Attribution-ShareAlike 3.0 Unported License
    (https://childes.talkbank.org/access/Eng-NA/Kuczaj.html):

  > Kuczaj, S. (1977). The acquisition of regular and irregular past tense
  > forms. _Journal of Verbal Learning and Verbal Behavior, 16_, 589–600.

- **Devil's Advocate**

  - The complete de-identified dataset of raw conversational transcripts
    is hosted on a secure protected-access repository provided by the
    Inter-university Consortium for Political and Social Research
    (ICPSR). Please click on the link to access: http://dx.doi.org/10.3886/ICPSR37124.v1.
    Due to the requirements of our IRB, please note that users interested in
    obtaining these data must complete a Restricted Data Use Agreement, specify
    the reason for the request, and obtain IRB approval or notice of exemption for their research.

  > Duran, Nicholas, Alexandra Paxton, and Riccardo
  > Fusaroli. Conversational Transcripts of Truthful and
  > Deceptive Speech Involving Controversial Topics,
  > Central California, 2012. ICPSR37124-v1. Ann Arbor,
  > MI: Inter-university Consortium for Political and
  > Social Research [distributor], 2018-08-29.
