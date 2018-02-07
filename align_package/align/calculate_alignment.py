import os,re,math,csv,string,random,logging,glob,itertools,operator, sys
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict, OrderedDict
from itertools import chain, combinations

import pandas as pd
import numpy as np
import scipy
from scipy import spatial

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from nltk.util import ngrams

from gensim.models import word2vec


def ngram_pos(sequence1,sequence2,ngramsize=2,
                   ignore_duplicates=True):
    """
    Remove mimicked lexical sequences from two interlocutors'
    sequences and return a dictionary of counts of ngrams
    of the desired size for each sequence.

    By default, consider bigrams. If desired, this may be
    changed by setting `ngramsize` to the appropriate
    value.

    By default, ignore duplicate lexical n-grams when
    processing these sequences. If desired, this may
    be changed with `ignore_duplicates=False`.
    """

    # remove duplicates and recreate sequences
    sequence1 = set(ngrams(sequence1,ngramsize))
    sequence2 = set(ngrams(sequence2,ngramsize))

    # if desired, remove duplicates from sequences
    if ignore_duplicates:
        new_sequence1 = [tuple([''.join(pair[1]) for pair in tup]) for tup in list(sequence1 - sequence2)]
        new_sequence2 = [tuple([''.join(pair[1]) for pair in tup]) for tup in list(sequence2 - sequence1)]
    else:
        new_sequence1 = [tuple([''.join(pair[1]) for pair in tup]) for tup in sequence1]
        new_sequence2 = [tuple([''.join(pair[1]) for pair in tup]) for tup in sequence2]

    # return counters
    return Counter(new_sequence1), Counter(new_sequence2)


def ngram_lexical(sequence1,sequence2,ngramsize=2):
    """
    Create ngrams of the desired size for each of two
    interlocutors' sequences and return a dictionary
    of counts of ngrams for each sequence.

    By default, consider bigrams. If desired, this may be
    changed by setting `ngramsize` to the appropriate
    value.
    """

    # generate ngrams
    sequence1 = list(ngrams(sequence1,ngramsize))
    sequence2 = list(ngrams(sequence2,ngramsize))

    # join for counters
    new_sequence1 = [' '.join(pair) for pair in sequence1]
    new_sequence2 = [' '.join(pair) for pair in sequence2]

    # return counters
    return Counter(new_sequence1), Counter(new_sequence2)


def get_cosine(vec1, vec2):
    """
    Derive cosine similarity metric, standard measure.
    Adapted from <https://stackoverflow.com/a/33129724>.
    """

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def build_composite_semantic_vector(lemma_seq,vocablist,highDimModel):
    """
    Function for producing vocablist and model is called in the main loop
    """

    ## filter out words in corpus that do not appear in vocablist (either too rare or too frequent)
    filter_lemma_seq = [word for word in lemma_seq if word in vocablist]
    ## build composite vector
    getComposite = [0] * len(highDimModel[vocablist[1]])
    for w1 in filter_lemma_seq:
        if w1 in highDimModel.vocab:
            semvector = highDimModel[w1]
            getComposite = getComposite + semvector
    return getComposite


def BuildSemanticModel(semantic_model_input_file,
                        pretrained_input_file,
                        use_pretrained_vectors=True,
                        high_sd_cutoff=3,
                        low_n_cutoff=1):

    """
    Given an input file produced by the ALIGN Phase 1 functions,
    build a semantic model from all transcripts in all conversations
    in target corpus after removing high- and low-frequency words.
    High-frequency words are determined by a user-defined number of
    SDs over the mean (by default, `high_sd_cutoff=3`). Low-frequency
    words must appear over a specified number of raw occurrences
    (by default, `low_n_cutoff=1`).

    Frequency cutoffs can be removed by `high_sd_cutoff=None` and/or
    `low_n_cutoff=0`.
    """

    # build vocabulary list from transcripts
    data1 = pd.read_csv(semantic_model_input_file, sep='\t', encoding='utf-8')

    # get frequency count of all included words
    all_sentences = [re.sub('[^\w\s]+','',str(row)).split(' ') for row in list(data1['lemma'])]
    all_words = list([a for b in all_sentences for a in b])
    frequency = defaultdict(int)
    for word in all_words:
        frequency[word] += 1

    # remove words that only occur more frequently than our cutoff (defined in occurrences)
    frequency = {word: freq for word, freq in frequency.iteritems() if freq > low_n_cutoff}

    # if desired, remove high-frequency words (over user-defined SDs above mean)
    if high_sd_cutoff is None:
        contentWords = [word for word in frequency.keys()]
    else:
        getOut = np.mean(frequency.values())+(np.std(frequency.values())*(high_sd_cutoff))
        contentWords = {word: freq for word, freq in frequency.iteritems() if freq < getOut}.keys()

    # decide whether to build semantic model from scratch or load in pretrained vectors
    if not use_pretrained_vectors:
        keepSentences = [[word for word in row if word in contentWords] for row in all_sentences]
        semantic_model = word2vec.Word2Vec(all_sentences, min_count=low_n_cutoff)
    else:
        if pretrained_input_file is None:
            raise ValueError('Error! Specify path to pretrained vector file using the `pretrained_input_file` argument.')
        else:
            semantic_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_input_file, binary=True)

    # return all the content words and the trained word vectors
    return contentWords, semantic_model.wv


def LexicalPOSAlignment(tok1,lem1,penn_tok1,penn_lem1,
                             tok2,lem2,penn_tok2,penn_lem2,
                             stan_tok1=None,stan_lem1=None,
                             stan_tok2=None,stan_lem2=None,
                             maxngram=2,
                             ignore_duplicates=True,
                             add_stanford_tags=False):

    """
    Derive lexical and part-of-speech alignment scores
    between interlocutors (suffix `1` and `2` in arguments
    passed to function).

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True` and by providing appropriate
    values for `stan_tok1`, `stan_lem1`, `stan_tok2`, and
    `stan_lem2`.

    By default, consider only bigram when calculating
    similarity. If desired, this window may be expanded
    by changing the `maxngram` argument value.

    By default, remove exact duplicates when calculating
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # create empty dictionaries for syntactic similarity
    syntax_penn_tok = {}
    syntax_penn_lem = {}

    # if desired, generate Stanford-based scores
    if add_stanford_tags:
        syntax_stan_tok = {}
        syntax_stan_lem = {}

    # create empty dictionaries for lexical similarity
    lexical_tok = {}
    lexical_lem = {}

    # cycle through all desired ngram lengths
    for ngram in range(2,maxngram+1):

        # calculate similarity for lexical ngrams (tokens and lemmas)
        [vectorT1, vectorT2] = ngram_lexical(tok1,tok2,ngramsize=ngram)
        [vectorL1, vectorL2] = ngram_lexical(lem1,lem2,ngramsize=ngram)
        lexical_tok['lexical_tok{0}'.format(ngram)] = get_cosine(vectorT1,vectorT2)
        lexical_lem['lexical_lem{0}'.format(ngram)] = get_cosine(vectorL1, vectorL2)

        # calculate similarity for Penn POS ngrams (tokens)
        [vector_penn_tok1, vector_penn_tok2] = ngram_pos(penn_tok1,penn_tok2,
                                                ngramsize=ngram,
                                                ignore_duplicates=ignore_duplicates)
        syntax_penn_tok['syntax_penn_tok{0}'.format(ngram)] = get_cosine(vector_penn_tok1,
                                                                                            vector_penn_tok2)
        # calculate similarity for Penn POS ngrams (lemmas)
        [vector_penn_lem1, vector_penn_lem2] = ngram_pos(penn_lem1,penn_lem2,
                                                              ngramsize=ngram,
                                                              ignore_duplicates=ignore_duplicates)
        syntax_penn_lem['syntax_penn_lem{0}'.format(ngram)] = get_cosine(vector_penn_lem1,
                                                                                            vector_penn_lem2)

        # if desired, also calculate using Stanford POS
        if add_stanford_tags:

            # calculate similarity for Stanford POS ngrams (tokens)
            [vector_stan_tok1, vector_stan_tok2] = ngram_pos(stan_tok1,stan_tok2,
                                                                  ngramsize=ngram,
                                                                  ignore_duplicates=ignore_duplicates)
            syntax_stan_tok['syntax_stan_tok{0}'.format(ngram)] = get_cosine(vector_stan_tok1,
                                                                                                vector_stan_tok2)

            # calculate similarity for Stanford POS ngrams (lemmas)
            [vector_stan_lem1, vector_stan_lem2] = ngram_pos(stan_lem1,stan_lem2,
                                                                  ngramsize=ngram,
                                                                  ignore_duplicates=ignore_duplicates)
            syntax_stan_lem['syntax_stan_lem{0}'.format(ngram)] = get_cosine(vector_stan_lem1,
                                                                                                vector_stan_lem2)

    # return requested information
    if add_stanford_tags:
        dictionaries_list = [syntax_penn_tok, syntax_penn_lem,
                             syntax_stan_tok, syntax_stan_lem,
                             lexical_tok, lexical_lem]
    else:
        dictionaries_list = [syntax_penn_tok, syntax_penn_lem,
                             lexical_tok, lexical_lem]

    return dictionaries_list


def conceptualAlignment(lem1, lem2, vocablist, highDimModel):

    """
    Calculate conceptual alignment scores from list of lemmas
    from between two interocutors (suffix `1` and `2` in arguments
    passed to function) using `word2vec`.
    """

    # aggregate composite high-dimensional vectors of all words in utterance
    W2Vec1 = build_composite_semantic_vector(lem1,vocablist,highDimModel)
    W2Vec2 = build_composite_semantic_vector(lem2,vocablist,highDimModel)

    # return cosine distance alignment score
    return 1 - spatial.distance.cosine(W2Vec1, W2Vec2)


def returnMultilevelAlignment(cond_info,
                                   partnerA,tok1,lem1,penn_tok1,penn_lem1,
                                   partnerB,tok2,lem2,penn_tok2,penn_lem2,
                                   vocablist, highDimModel,
                                   stan_tok1=None,stan_lem1=None,
                                   stan_tok2=None,stan_lem2=None,
                                   add_stanford_tags=False,
                                   maxngram=2,
                                   ignore_duplicates=True):

    """
    Calculate lexical, syntactic, and conceptual alignment
    between a pair of turns by individual interlocutors
    (suffix `1` and `2` in arguments passed to function),
    including leading/following comparison directionality.

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True` and by providing appropriate
    values for `stan_tok1`, `stan_lem1`, `stan_tok2`, and
    `stan_lem2`.

    By default, consider only bigrams when calculating
    similarity. If desired, this window may be expanded
    by changing the `maxngram` argument value.

    By default, remove exact duplicates when calculating
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # create empty dictionaries
    partner_direction = {}
    condition_info = {}
    cosine_semanticL = {}

    # calculate lexical and syntactic alignment
    dictionaries_list = LexicalPOSAlignment(tok1=tok1,lem1=lem1,
                                                 penn_tok1=penn_tok1,penn_lem1=penn_lem1,
                                                 tok2=tok2,lem2=lem2,
                                                 penn_tok2=penn_tok2,penn_lem2=penn_lem2,
                                                 stan_tok1=stan_tok1,stan_lem1=stan_lem1,
                                                 stan_tok2=stan_tok2,stan_lem2=stan_lem2,
                                                 maxngram=maxngram,
                                                 ignore_duplicates=ignore_duplicates,
                                                 add_stanford_tags=add_stanford_tags)

    # calculate conceptual alignment
    cosine_semanticL['cosine_semanticL'] = conceptualAlignment(lem1,lem2,vocablist,highDimModel)
    dictionaries_list.append(cosine_semanticL.copy())

    # determine directionality of leading/following comparison
    partner_direction['partner_direction'] = str(partnerA) + ">" + str(partnerB)
    dictionaries_list.append(partner_direction.copy())

    # add condition information
    condition_info['condition_info'] = cond_info
    dictionaries_list.append(condition_info.copy())

    # return alignment scores
    return dictionaries_list


def TurnByTurnAnalysis(dataframe,
                            vocablist,
                            highDimModel,
                            delay=1,
                            maxngram=2,
                            add_stanford_tags=False,
                            ignore_duplicates=True):

    """
    Calculate lexical, syntactic, and conceptual alignment
    between interlocutors over an entire conversation.
    Automatically detect individual speakers by unique
    speaker codes.

    By default, compare only adjacent turns. If desired,
    the comparison distance may be changed by increasing
    the `delay` argument.

    By default, include maximum n-gram comparison of 2. If
    desired, this may be changed by passing the appropriate
    value to the the `maxngram` argument.

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True`.

    By default, remove exact duplicates when calculating POS
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # if we don't want the Stanford tagger data, set defaults
    if not add_stanford_tags:
        stan_tok1=None
        stan_lem1=None
        stan_tok2=None
        stan_lem2=None

    # prepare the data to the appropriate type
    dataframe['token'] = dataframe['token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['lemma'] = dataframe['lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['tagged_token'] = dataframe['tagged_token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['tagged_token'] = dataframe['tagged_token'].apply(lambda x: zip(x[0::2],x[1::2])) # thanks to https://stackoverflow.com/a/4647086
    dataframe['tagged_lemma'] = dataframe['tagged_lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
    dataframe['tagged_lemma'] = dataframe['tagged_lemma'].apply(lambda x: zip(x[0::2],x[1::2])) # thanks to https://stackoverflow.com/a/4647086

    # if desired, prepare the Stanford tagger data
    if add_stanford_tags:
        dataframe['tagged_stan_token'] = dataframe['tagged_stan_token'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
        dataframe['tagged_stan_token'] = dataframe['tagged_stan_token'].apply(lambda x: zip(x[0::2],x[1::2])) # thanks to https://stackoverflow.com/a/4647086
        dataframe['tagged_stan_lemma'] = dataframe['tagged_stan_lemma'].apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
        dataframe['tagged_stan_lemma'] = dataframe['tagged_stan_lemma'].apply(lambda x: zip(x[0::2],x[1::2])) # thanks to https://stackoverflow.com/a/4647086

    # create lagged version of the dataframe
    df_original = dataframe.drop(dataframe.tail(delay).index,inplace=False)
    df_lagged = dataframe.shift(-delay).drop(dataframe.tail(delay).index,inplace=False)

    # cycle through each pair of turns
    aggregated_df = pd.DataFrame()
    for i in range(0,df_original.shape[0]):

        # identify the condition for this dataframe
        cond_info = dataframe['file'].unique()
        if len(cond_info)==1:
            cond_info = str(cond_info[0])

        # break and flag error if we have more than 1 condition per dataframe
        else:
            raise ValueError('Error! Dataframe contains multiple conditions. Split dataframe into multiple dataframes, one per condition: '+cond_info)

        # grab all of first participant's data
        first_row = df_original.iloc[i]
        first_partner = first_row['participant']
        tok1=first_row['token']
        lem1=first_row['lemma']
        penn_tok1=first_row['tagged_token']
        penn_lem1=first_row['tagged_lemma']

        # grab all of lagged participant's data
        lagged_row = df_lagged.iloc[i]
        lagged_partner = lagged_row['participant']
        tok2=lagged_row['token']
        lem2=lagged_row['lemma']
        penn_tok2=lagged_row['tagged_token']
        penn_lem2=lagged_row['tagged_lemma']

        # if desired, grab the Stanford tagger data for both participants
        if add_stanford_tags:
            stan_tok1=first_row['tagged_stan_token']
            stan_lem1=first_row['tagged_stan_lemma']
            stan_tok2=lagged_row['tagged_stan_token']
            stan_lem2=lagged_row['tagged_stan_lemma']

        # process multilevel alignment
        dictionaries_list=returnMultilevelAlignment(cond_info=cond_info,
                                                         partnerA=first_partner,
                                                         tok1=tok1,lem1=lem1,
                                                         penn_tok1=penn_tok1,penn_lem1=penn_lem1,
                                                         partnerB=lagged_partner,
                                                         tok2=tok2,lem2=lem2,
                                                         penn_tok2=penn_tok2,penn_lem2=penn_lem2,
                                                         vocablist=vocablist,
                                                         highDimModel=highDimModel,
                                                         stan_tok1=stan_tok1,stan_lem1=stan_lem1,
                                                         stan_tok2=stan_tok2,stan_lem2=stan_lem2,
                                                         maxngram = maxngram,
                                                         ignore_duplicates = ignore_duplicates,
                                                         add_stanford_tags = add_stanford_tags)

        # sort columns so they are in order, append data to existing structures
        next_df_line = pd.DataFrame.from_dict(OrderedDict(k for num, i in enumerate(d for d in dictionaries_list) for k in sorted(i.items())),
                               orient='index').transpose()
        aggregated_df = aggregated_df.append(next_df_line)

    # reformat turn information and add index
    aggregated_df = aggregated_df.reset_index(drop=True).reset_index().rename(columns={"index":"time"})

    # give us our finished dataframe
    return aggregated_df


def ConvoByConvoAnalysis(dataframe,
                          maxngram=2,
                          ignore_duplicates=True,
                          add_stanford_tags=False):

    """
    Calculate analysis of multilevel similarity over
    a conversation between two interlocutors from a
    transcript dataframe prepared by Phase 1
    of ALIGN. Automatically detect speakers by unique
    speaker codes.

    By default, include maximum n-gram comparison of 2. If
    desired, this may be changed by passing the appropriate
    value to the the `maxngram` argument.

    By default, return scores based only on Penn POS taggers.
    If desired, also return scores using Stanford tagger with
    `add_stanford_tags=True`.

    By default, remove exact duplicates when calculating POS
    similarity scores (i.e., does not consider perfectly
    mimicked lexical items between speakers). If desired,
    duplicates may be included when calculating scores by
    passing `ignore_duplicates=False`.
    """

    # identify the condition for this dataframe
    cond_info = dataframe['file'].unique()
    if len(cond_info)==1:
        cond_info = str(cond_info[0])

    # break and flag error if we have more than 1 condition per dataframe
    else:
        raise ValueError('Error! Dataframe contains multiple conditions. Split dataframe into multiple dataframes, one per condition: '+cond_info)

    # if we don't want the Stanford info, set defaults
    if not add_stanford_tags:
        stan_tok1 = None
        stan_lem1 = None
        stan_tok2 = None
        stan_lem2 = None

    # identify individual interlocutors
    df_A = dataframe.loc[dataframe['participant'] == dataframe['participant'].unique()[0]]
    df_B = dataframe.loc[dataframe['participant'] == dataframe['participant'].unique()[1]]

    # concatenate the token, lemma, and POS information for participant A
    tok1 = [word for turn in df_A['token'] for word in turn]
    lem1 = [word for turn in df_A['lemma'] for word in turn]
    penn_tok1 = [POS for turn in df_A['tagged_token'] for POS in turn]
    penn_lem1 = [POS for turn in df_A['tagged_token'] for POS in turn]
    if add_stanford_tags:

        if isinstance(df_A['tagged_stan_token'][0], list):
            stan_tok1 = [POS for turn in df_A['tagged_stan_token'] for POS in turn]
            stan_lem1 = [POS for turn in df_A['tagged_stan_lemma'] for POS in turn]

        elif isinstance(df_A['tagged_stan_token'][0], unicode):
            stan_tok1 = pd.Series(df_A['tagged_stan_token'].values).apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
            stan_tok1 = stan_tok1.apply(lambda x: zip(x[0::2],x[1::2]))
            stan_tok1 = [POS for turn in stan_tok1 for POS in turn]
            stan_lem1 = pd.Series(df_A['tagged_stan_lemma'].values).apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
            stan_lem1 = stan_lem1.apply(lambda x: zip(x[0::2],x[1::2]))
            stan_lem1 = [POS for turn in stan_lem1 for POS in turn]

    # concatenate the token, lemma, and POS information for participant B
    tok2 = [word for turn in df_B['token'] for word in turn]
    lem2 = [word for turn in df_B['lemma'] for word in turn]
    penn_tok2 = [POS for turn in df_B['tagged_token'] for POS in turn]
    penn_lem2 = [POS for turn in df_B['tagged_token'] for POS in turn]
    if add_stanford_tags:

        if isinstance(df_A['tagged_stan_token'][0],list):
            stan_tok2 = [POS for turn in df_B['tagged_stan_token'] for POS in turn]
            stan_lem2 = [POS for turn in df_B['tagged_stan_lemma'] for POS in turn]

        elif isinstance(df_A['tagged_stan_token'][0], unicode):
            stan_tok2 = pd.Series(df_B['tagged_stan_token'].values).apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
            stan_tok2 = stan_tok2.apply(lambda x: zip(x[0::2],x[1::2]))
            stan_tok2 = [POS for turn in stan_tok2 for POS in turn]
            stan_lem2 = pd.Series(df_B['tagged_stan_lemma'].values).apply(lambda x: re.sub('[^\w\s]+','',x).split(' '))
            stan_lem2 = stan_lem2.apply(lambda x: zip(x[0::2],x[1::2]))
            stan_lem2 = [POS for turn in stan_lem2 for POS in turn]

    # process multilevel alignment
    dictionaries_list = LexicalPOSAlignment(tok1=tok1,lem1=lem1,
                                                 penn_tok1=penn_tok1,penn_lem1=penn_lem1,
                                                 tok2=tok2,lem2=lem2,
                                                 penn_tok2=penn_tok2,penn_lem2=penn_lem2,
                                                 stan_tok1=stan_tok1,stan_lem1=stan_lem1,
                                                 stan_tok2=stan_tok2,stan_lem2=stan_lem2,
                                                 maxngram=maxngram,
                                                 ignore_duplicates=ignore_duplicates,
                                                 add_stanford_tags=add_stanford_tags)

    # append data to existing structures
    dictionary_df = pd.DataFrame.from_dict(OrderedDict(k for num, i in enumerate(d for d in dictionaries_list) for k in sorted(i.items())),
                       orient='index').transpose()
    dictionary_df['condition_info'] = cond_info

    # return the dataframe
    return dictionary_df


def GenerateSurrogate(original_conversation_list,
                           surrogate_file_directory,
                           all_surrogates=True,
                           keep_original_turn_order=True,
                           id_separator = '\-',
                           dyad_label='dyad',
                           condition_label='cond'):

    """
    Create transcripts for surrogate pairs of
    participants (i.e., participants who did not
    genuinely interact in the experiment), which
    will later be used to generate baseline levels
    of alignment. Store surrogate files in a new
    folder each time the surrogate generation is run.

    Returns a list of all surrogate files created.

    By default, the separator between dyad ID and
    condition ID is a hyphen ('\-'). If desired,
    this may be changed in the `id_separator`
    argument.

    By default, condition IDs will be identified as
    any characters following `cond`. If desired,
    this may be changed with the `condition_label`
    argument.

    By default, dyad IDs will be identified as
    any characters following `dyad`. If desired,
    this may be changed with the `dyad_label`
    argument.

    By default, generate surrogates from all possible
    pairings. If desired, instead generate surrogates
    only from a subset of all possible pairings
    with `all_surrogates=False`.

    By default, create surrogates by retaining the
    original ordering of each surrogate partner's
    data. If desired, create surrogates by shuffling
    all turns within each surrogate partner's data
    with `keep_original_turn_order = False`.
    """

    # create a subfolder for the new set of surrogates
    import time
    new_surrogate_path = surrogate_file_directory + 'surrogate_run-' + str(time.time()) +'/'
    if not os.path.exists(new_surrogate_path):
        os.makedirs(new_surrogate_path)

    # grab condition types from each file name
    file_info = [re.sub('\.txt','',os.path.basename(file_name)) for file_name in original_conversation_list]
    condition_ids = list(set([re.findall('[^'+id_separator+']*'+condition_label+'.*',metadata)[0] for metadata in file_info]))
    files_conditions = {}
    for unique_condition in condition_ids:
        next_condition_files = [add_file for add_file in original_conversation_list if unique_condition in add_file]
        files_conditions[unique_condition] = next_condition_files

    # cycle through conditions
    for condition in files_conditions.keys():

        # default: grab all possible pairs of conversations of this condition
        paired_surrogates = [pair for pair in combinations(files_conditions[condition],2)]

        # otherwise, if desired, randomly pull from all pairs to get target surrogate sample
        if not all_surrogates:
            import math
            paired_surrogates = random.sample(paired_surrogates,
                                              int(math.ceil(len(files_conditions[condition])/2)))

        # cycle through surrogate pairings
        for next_surrogate in paired_surrogates:

            # read in the files
            original_file1 = os.path.basename(next_surrogate[0])
            original_file2 = os.path.basename(next_surrogate[1])
            original_df1=pd.read_csv(next_surrogate[0], sep='\t',encoding='utf-8')
            original_df2=pd.read_csv(next_surrogate[1], sep='\t',encoding='utf-8')

            # get participants A and B from df1
            participantA_1_code = min(original_df1['participant'].unique())
            participantB_1_code = max(original_df1['participant'].unique())
            participantA_1 = original_df1[original_df1['participant'] == participantA_1_code].reset_index().rename(columns={'file': 'original_file'})
            participantB_1 = original_df1[original_df1['participant'] == participantB_1_code].reset_index().rename(columns={'file': 'original_file'})

            # get participants A and B from df2
            participantA_2_code = min(original_df2['participant'].unique())
            participantB_2_code = max(original_df2['participant'].unique())
            participantA_2 = original_df2[original_df2['participant'] == participantA_2_code].reset_index().rename(columns={'file': 'original_file'})
            participantB_2 = original_df2[original_df2['participant'] == participantB_2_code].reset_index().rename(columns={'file': 'original_file'})

            # identify truncation point for both surrogates (to have even number of turns)
            surrogateX_turns=min([participantA_1.shape[0],
                                  participantB_2.shape[0]])
            surrogateY_turns=min([participantA_2.shape[0],
                                  participantB_1.shape[0]])

            # preserve original turn order for surrogate pairs
            if keep_original_turn_order:
                surrogateX_A1 = participantA_1.truncate(after=surrogateX_turns-1,
                                                        copy=False)
                surrogateX_B2 = participantB_2.truncate(after=surrogateX_turns-1,
                                                        copy=False)
                surrogateX = pd.concat(
                    [surrogateX_A1, surrogateX_B2]).sort_index(
                            kind="mergesort").reset_index(
                                    drop=True).rename(
                                        columns={'index': 'original_index'})

                surrogateY_A2 = participantA_2.truncate(after=surrogateY_turns-1,
                                                        copy=False)
                surrogateY_B1 = participantB_1.truncate(after=surrogateY_turns-1,
                                                        copy=False)
                surrogateY = pd.concat(
                    [surrogateY_A2, surrogateY_B1]).sort_index(
                            kind="mergesort").reset_index(
                                    drop=True).rename(
                                            columns={'index': 'original_index'})

            # otherwise, if desired, just shuffle all turns within participants
            else:

                # shuffle for first surrogate pairing
                surrogateX_A1 = participantA_1.truncate(after=surrogateX_turns-1,copy=False).sample(frac=1).reset_index(drop=True)
                surrogateX_B2 = participantB_2.truncate(after=surrogateX_turns-1,copy=False).sample(frac=1).reset_index(drop=True)
                surrogateX = pd.concat([surrogateX_A1,surrogateX_B2]).sort_index(kind="mergesort").reset_index(drop=True).rename(columns={'index': 'original_index'})

                # and for second surrogate pairing
                surrogateY_A2 = participantA_2.truncate(after=surrogateY_turns-1,copy=False).sample(frac=1).reset_index(drop=True)
                surrogateY_B1 = participantB_1.truncate(after=surrogateY_turns-1,copy=False).sample(frac=1).reset_index(drop=True)
                surrogateY = pd.concat([surrogateY_A2,surrogateY_B1]).sort_index(kind="mergesort").reset_index(drop=True).rename(columns={'index': 'original_index'})

            # create filename for our surrogate file
            original_dyad1 = re.findall(dyad_label+'[^'+id_separator+']*',original_file1)[0]
            original_dyad2 = re.findall(dyad_label+'[^'+id_separator+']*',original_file2)[0]
            surrogateX['file'] = original_dyad1 + '-' + original_dyad2 + '-' + condition
            surrogateY['file'] = original_dyad2 + '-' + original_dyad1 + '-' + condition
            nameX='SurrogatePair-'+original_dyad1+'A'+'-'+original_dyad2+'B'+'-'+condition+'.txt'
            nameY='SurrogatePair-'+original_dyad2+'A'+'-'+original_dyad1+'B'+'-'+condition+'.txt'

            # save to file
            surrogateX.to_csv(new_surrogate_path + nameX, encoding='utf-8',index=False,sep='\t')
            surrogateY.to_csv(new_surrogate_path + nameY, encoding='utf-8',index=False,sep='\t')

    # return list of all surrogate files
    return glob.glob(new_surrogate_path+"*.txt")


def calculate_alignment(input_files,
                        output_file_directory,
                        semantic_model_input_file,
                        pretrained_input_file,
                        high_sd_cutoff=3,
                        low_n_cutoff=1,
                        delay=1,
                        maxngram=2,
                        use_pretrained_vectors=True,
                        ignore_duplicates=True,
                        add_stanford_tags=False,
                        input_as_directory=True):

    """
    Calculate lexical, syntactic, and conceptual alignment between speakers.

    Given a directory of individual .txt files and the
    vocabulary list that have been generated by the `prepare_transcripts`
    preparation stage, return multi-level alignment
    scores with turn-by-turn and conversation-level metrics.

    Parameters
    ----------

    input_files : str (directory name) or list of str (file names)
        Cleaned files to be analyzed. Behavior governed by `input_as_directory`
        parameter as well.

    output_file_directory : str
        Name of directory where output for individual conversations will be
        saved.

    semantic_model_input_file : str
        Name of file to be used for creating the semantic model. A compatible
        file will be saved as an output of `prepare_transcripts()`.

    pretrained_input_file : str or None
        If using a pretrained vector to create the semantic model, use
        name of model here. If not, use None. Behavior governed by
        `use_pretrained_vectors` parameter as well.

    high_sd_cutoff : int, optional (default: 3)
        High-frequency cutoff (in SD over the mean) for lexical items
        when creating the semantic model.

    low_n_cutoff : int, optional (default: 1)
        Low-frequency cutoff (in raw frequency) for lexical items when
        creating the semantic models. Items with frequency less than or
        equal to the number provided here will be removed. To remove the
        low-frequency cutoff, set to 0.

    delay : int, optional (default: 1)
        Delay (or lag) at which to calculate similarity. A lag of 1 (default)
        considers only adjacent turns.

    maxngram : int, optional (default: 2)
        Maximum n-gram size for calculations. Similarity scores for n-grams
        from unigrams to the maximum size specified here will be calculated.

    use_pretrained_vectors : boolean, optional (default: True)
        Specify whether to use a pretrained gensim model for word2vec
        analysis (True) or to construct a new model from the provided corpus
        (False). If True, the file name of a valid model must be
        provided to the `pretrained_input_file` parameter.

    ignore_duplicates : boolean, optional (default: True)
        Specify whether to remove exact duplicates when calculating
        part-of-speech similarity scores (True) or to retain perfectly
        mimicked lexical items for POS similarity calculation (False).

    add_stanford_tags : boolean, optional (default: False)
        Specify whether to return part-of-speech similarity scores based on
        Stanford POS tagger in addition to the Penn POS tagger (True) or to
        return only POS similarity scores from the Penn tagger (False).

    input_as_directory : boolean, optional (default: True)
        Specify whether the value passed to `input_files` parameter should
        be read as a directory (True) or a list of files to be processed
        (False).
    """

    # grab the files in the list
    if not input_as_directory:
        file_list = glob.glob(input_files)
    else:
        file_list = glob.glob(input_files+"/*.txt")

    # build the semantic model to be used for all conversations
    [vocablist, highDimModel] = BuildSemanticModel(semantic_model_input_file=semantic_model_input_file,
                                                       pretrained_input_file=pretrained_input_file,
                                                       use_pretrained_vectors=use_pretrained_vectors,
                                                       high_sd_cutoff=high_sd_cutoff,
                                                       low_n_cutoff=low_n_cutoff)

    # create containers for alignment values
    AlignmentT2T = pd.DataFrame()
    AlignmentC2C = pd.DataFrame()

    # cycle through each prepared file
    for fileName in file_list:

        # process the file if it's got a valid conversation
        dataframe=pd.read_csv(fileName, sep='\t',encoding='utf-8')
        if len(dataframe) > 1:

            # let us know which filename we're processing
            print "Processing: "+fileName

            # calculate turn-by-turn alignment scores
            xT2T=TurnByTurnAnalysis(dataframe=dataframe,
                                         delay=delay,
                                         maxngram=maxngram,
                                         vocablist=vocablist,
                                         highDimModel=highDimModel,
                                         add_stanford_tags=add_stanford_tags,
                                         ignore_duplicates=ignore_duplicates)
            AlignmentT2T=AlignmentT2T.append(xT2T)

            # calculate conversation-level alignment scores
            xC2C = ConvoByConvoAnalysis(dataframe=dataframe,
                                             maxngram = maxngram,
                                             ignore_duplicates=ignore_duplicates,
                                             add_stanford_tags = add_stanford_tags)
            AlignmentC2C=AlignmentC2C.append(xC2C)

        # if it's invalid, let us know
        else:
            print "Invalid file: "+fileName

    # update final dataframes
    FINAL_TURN = AlignmentT2T.reset_index(drop=True)
    FINAL_CONVO = AlignmentC2C.reset_index(drop=True)

    # export the final files
    FINAL_TURN.to_csv(output_file_directory+"AlignmentT2T.txt",
                      encoding='utf-8', index=False, sep='\t')
    FINAL_CONVO.to_csv(output_file_directory+"AlignmentC2C.txt",
                       encoding='utf-8', index=False, sep='\t')

    # display the info, too
    return FINAL_TURN, FINAL_CONVO


def calculate_baseline_alignment(input_files,
                                 surrogate_file_directory,
                                 output_file_directory,
                                 semantic_model_input_file,
                                 pretrained_input_file,
                                 high_sd_cutoff=3,
                                 low_n_cutoff=1,
                                 id_separator='\-',
                                 condition_label='cond',
                                 dyad_label='dyad',
                                 all_surrogates=True,
                                 keep_original_turn_order=True,
                                 delay=1,
                                 maxngram=2,
                                 use_pretrained_vectors=True,
                                 ignore_duplicates=True,
                                 add_stanford_tags=False,
                                 input_as_directory=True):

    """
    Calculate baselines for lexical, syntactic, and conceptual
    alignment between speakers.

    Given a directory of individual .txt files and the
    vocab list that have been generated by the `prepare_transcripts`
    preparation stage, return multi-level alignment
    scores with turn-by-turn and conversation-level metrics
    for surrogate baseline conversations.

    Parameters
    ----------

    input_files : str (directory name) or list of str (file names)
        Cleaned files to be analyzed. Behavior governed by `input_as_directory`
        parameter as well.

    surrogate_file_directory : str
        Name of directory where raw surrogate data will be saved.

    output_file_directory : str
        Name of directory where output for individual surrogate
        conversations will be saved.

    semantic_model_input_file : str
        Name of file to be used for creating the semantic model. A compatible
        file will be saved as an output of `prepare_transcripts()`.

    pretrained_input_file : str or None
        If using a pretrained vector to create the semantic model, use
        name of model here. If not, use None. Behavior governed by
        `use_pretrained_vectors` parameter as well.

    high_sd_cutoff : int, optional (default: 3)
        High-frequency cutoff (in SD over the mean) for lexical items
        when creating the semantic model.

    low_n_cutoff : int, optional (default: 1)
        Low-frequency cutoff (in raw frequency) for lexical items when
        creating the semantic models. Items with frequency less than or
        equal to the number provided here will be removed. To remove the
        low-frequency cutoff, set to 0.

    id_separator : str, optional (default: '\-')
        Character separator between the dyad and condition IDs in
        original data file names.

    condition_label : str, optional (default: 'cond')
        String preceding ID for each unique condition. Anything after this
        label will be identified as a unique condition ID.

    dyad_label : str, optional (default: 'dyad')
        String preceding ID for each unique dyad. Anything after this label
        will be identified as a unique dyad ID.

    all_surrogates : boolean, optional (default: True)
        Specify whether to generate all possible surrogates across original
        dataset (True) or to generate only a subset of surrogates equal to
        the real sample size drawn randomly from all possible surrogates
        (False).

    keep_original_turn_order : boolean, optional (default: True)
        Specify whether to retain original turn ordering when pairing surrogate
        dyads (True) or to pair surrogate partners' turns in random order
        (False).

    delay : int, optional (default: 1)
        Delay (or lag) at which to calculate similarity. A lag of 1 (default)
        considers only adjacent turns.

    maxngram : int, optional (default: 2)
        Maximum n-gram size for calculations. Similarity scores for n-grams
        from unigrams to the maximum size specified here will be calculated.

    use_pretrained_vectors : boolean, optional (default: True)
        Specify whether to use a pretrained gensim model for word2vec
        analysis. If True, the file name of a valid model must be
        provided to the `pretrained_input_file` parameter.

    ignore_duplicates : boolean, optional (default: True)
        Specify whether to remove exact duplicates when calculating
        part-of-speech similarity scores. By default, ignore perfectly
        mimicked lexical items for POS similarity calculation.

    add_stanford_tags : boolean, optional (default: False)
        Specify whether to return part-of-speech similarity scores
        based on Stanford POS tagger (in addition to the Penn POS
        tagger).

    input_as_directory : boolean, optional (default: True)
        Specify whether the value passed to `input_files` parameter should
        be read as a directory or a list of files to be processed.
    """

    # grab the files in the input list
    if not input_as_directory:
        file_list = glob.glob(input_files)
    else:
        file_list = glob.glob(input_files+"/*.txt")

    # create a surrogate file list
    surrogate_file_list = GenerateSurrogate(
                            original_conversation_list=file_list,
                            surrogate_file_directory=surrogate_file_directory,
                            all_surrogates=all_surrogates,
                            id_separator=id_separator,
                            condition_label=condition_label,
                            dyad_label=dyad_label,
                            keep_original_turn_order=keep_original_turn_order)

    # build the semantic model to be used for all conversations
    [vocablist, highDimModel] = BuildSemanticModel(
                            semantic_model_input_file=semantic_model_input_file,
                            pretrained_input_file=pretrained_input_file,
                            use_pretrained_vectors=use_pretrained_vectors,
                            high_sd_cutoff=high_sd_cutoff,
                            low_n_cutoff=low_n_cutoff)

    # create containers for alignment values
    AlignmentT2T = pd.DataFrame()
    AlignmentC2C = pd.DataFrame()

    # cycle through the files
    for fileName in surrogate_file_list:

        # process the file if it's got a valid conversation
        dataframe=pd.read_csv(fileName, sep='\t',encoding='utf-8')
        if len(dataframe) > 1:

            # let us know which filename we're processing
            print "Processing: "+fileName

            # calculate turn-by-turn alignment scores
            xT2T=TurnByTurnAnalysis(dataframe=dataframe,
                                         delay=delay,
                                         maxngram=maxngram,
                                         vocablist=vocablist,
                                         highDimModel=highDimModel,
                                         add_stanford_tags = add_stanford_tags,
                                         ignore_duplicates = ignore_duplicates)
            AlignmentT2T=AlignmentT2T.append(xT2T)

            # calculate conversation-level alignment scores
            xC2C = ConvoByConvoAnalysis(dataframe=dataframe,
                                             maxngram = maxngram,
                                             ignore_duplicates=ignore_duplicates,
                                             add_stanford_tags = add_stanford_tags)
            AlignmentC2C=AlignmentC2C.append(xC2C)

        # if it's invalid, let us know
        else:
            print "Invalid file: "+fileName

    # update final dataframes
    FINAL_TURN_SURROGATE = AlignmentT2T.reset_index(drop=True)
    FINAL_CONVO_SURROGATE = AlignmentC2C.reset_index(drop=True)

    # export the final files
    FINAL_TURN_SURROGATE.to_csv(output_file_directory+"AlignmentT2T_Surrogate.txt",
                      encoding='utf-8',index=False,sep='\t')
    FINAL_CONVO_SURROGATE.to_csv(output_file_directory+"AlignmentC2C_Surrogate.txt",
                       encoding='utf-8',index=False,sep='\t')

    # display the info, too
    return FINAL_TURN_SURROGATE, FINAL_CONVO_SURROGATE
