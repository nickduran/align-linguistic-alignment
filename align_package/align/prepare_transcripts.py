import os,re,math,csv,string,random,logging,glob,itertools,operator, sys
from os import listdir
from os.path import isfile, join
from collections import Counter, defaultdict, OrderedDict
from itertools import chain, combinations

import pandas as pd
import numpy as np
import scipy
from scipy import spatial

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag.stanford import StanfordPOSTagger
from nltk.util import ngrams

from gensim.models import word2vec

def InitialCleanup(dataframe,
                   minwords=2,
                   use_filler_list=None,
                   filler_regex_and_list=False):

    """
    Perform basic text cleaning to prepare dataframe
    for analysis. Remove non-letter/-space characters,
    empty turns, turns below a minimum length, and
    fillers.

    By default, preserves turns 2 words or longer.
    If desired, this may be changed by updating the
    `minwords` argument.

    By default, remove common fillers through regex.
    If desired, remove other words by passing a list
    of literal strings to `use_filler_list` argument,
    and if both regex and list of additional literal
    strings are to be used, update `filler_regex_and_list=True`.
    """

    # only allow strings, spaces, and newlines to pass
    WHITELIST = string.letters + '\'' + ' '

    # remove inadvertent empty turns
    dataframe = dataframe[pd.notnull(dataframe['content'])]

    # internal function: remove fillers via regular expressions
    def applyRegExpression(textFiller):
        textClean = re.sub('^(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]+\s', ' ', textFiller) # at the start of a string
        textClean = re.sub('\s(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]+\s', ' ', textClean) # within a string
        textClean = re.sub('\s(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]$', ' ', textClean) # end of a string
        textClean = re.sub('^(?!mom|am|ham)[u*|h*|m*|o*|a*]+[m*|h*|u*|a*]$', ' ', textClean) # if entire turn string
        return textClean

    # create a new column with only approved text before cleaning per user-specified settings
    dataframe['clean_content'] = dataframe['content'].apply(lambda utterance: ''.join([char for char in utterance if char in WHITELIST]).lower())

    # DEFAULT: remove typical speech fillers via regular expressions (examples: "um, mm, oh, hm, uh, ha")
    if use_filler_list is None and not filler_regex_and_list:
        dataframe['clean_content'] = dataframe['clean_content'].apply(applyRegExpression)

    # OPTION 1: remove speech fillers or other words specified by user in a list
    elif use_filler_list is not None and not filler_regex_and_list:
        dataframe['clean_content'] = dataframe['clean_content'].apply(lambda utterance: ' '.join([word for word in utterance.split(" ") if word not in use_filler_list]))

    # OPTION 2: remove speech fillers via regular expression and any additional words from user-specified list
    elif use_filler_list is not None and filler_regex_and_list:
        dataframe['clean_content'] = dataframe['clean_content'].apply(applyRegExpression)
        dataframe['clean_content'] = dataframe['clean_content'].apply(lambda utterance: ' '.join([word for word in utterance.split(" ") if word not in use_filler_list]))
        cleantext = " ".join(cleantext)

    # OPTION 3: nothing is filtered
    else:
        dataframe['clean_content'] = dataframe['clean_content']

    # drop the old "content" column and rename the clean "content" column
    dataframe = dataframe.drop(['content'],axis=1)
    dataframe = dataframe.rename(index=str,
                                 columns ={'clean_content': 'content'})

    # remove rows that are now blank or do not meet `minwords` requirement, then drop length column
    dataframe['utteranceLen'] = dataframe['content'].apply(lambda x: word_tokenize(x)).str.len()
    dataframe = dataframe.drop(dataframe[dataframe.utteranceLen < int(minwords)].index).drop(['utteranceLen'],axis=1)
    dataframe = dataframe.reset_index(drop=True)

    # return the cleaned dataframe
    return dataframe

def AdjacentMerge(dataframe):

    """
    Given a dataframe of conversation turns,
    merge adjacent turns by the same speaker.
    """

    repeat=1
    while repeat==1:
        l1=len(dataframe)
        DfMerge = []
        k = 0
        if len(dataframe) > 0:
            while k < len(dataframe)-1:
                if dataframe['participant'].iloc[k] != dataframe['participant'].iloc[k+1]:
                    DfMerge.append([dataframe['participant'].iloc[k], dataframe['content'].iloc[k]])
                    k = k + 1
                elif dataframe['participant'].iloc[k] == dataframe['participant'].iloc[k+1]:
                    DfMerge.append([dataframe['participant'].iloc[k], dataframe['content'].iloc[k] + " " + dataframe['content'].iloc[k+1]])
                    k = k + 2
            if k == len(dataframe)-1:
                DfMerge.append([dataframe['participant'].iloc[k], dataframe['content'].iloc[k]])

        dataframe=pd.DataFrame(DfMerge,columns=('participant','content'))
        if l1==len(dataframe):
            repeat=0

    return dataframe

def Tokenize(text,nwords):
    """
    Given list of text to be processed and a list
    of known words, return a list of edited and
    tokenized words.

    Spell-checking is implemented using a
    Bayesian spell-checking algorithm
    (http://norvig.com/spell-correct.html).

    By default, this is based on the Project Gutenberg
    corpus, a collection of approximately 1 million texts
    (http://www.gutenberg.org). A copy of this is included
    within this package. If desired, users may specify a
    different spell-check training corpus in the
    `training_dictionary` argument of the
    `prepare_transcripts()` function.

    """

    # internal function: identify possible spelling errors for a given word
    def edits1(word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in string.lowercase if b]
        inserts    = [a + c + b     for a, b in splits for c in string.lowercase]
        return set(deletes + transposes + replaces + inserts)

    # internal function: identify known edits
    def known_edits2(word,nwords):
        return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in nwords)

    # internal function: identify known words
    def known(words,nwords): return set(w for w in words if w in nwords)

    # internal function: correct spelling
    def correct(word,nwords):
        candidates = known([word],nwords) or known(edits1(word),nwords) or known_edits2(word,nwords) or [word]
        return max(candidates, key=nwords.get)

    # expand out based on a fixed list of common contractions
    contract_dict = { "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that had",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have" }
    contractions_re = re.compile('(%s)' % '|'.join(contract_dict.keys()))

    # internal function:
    def expand_contractions(text, contractions_re=contractions_re):
        def replace(match):
            return contract_dict[match.group(0)]
        return contractions_re.sub(replace, text.lower())

    # process all words in the text
    cleantoken = []
    text = expand_contractions(text)
    token = word_tokenize(text)
    for word in token:
        if "'" not in word:
            cleantoken.append(correct(word,nwords))
        else:
            cleantoken.append(word)
    return cleantoken

def pos_to_wn(tag):
    """
    Convert NLTK default tagger output into a format that Wordnet
    can use in order to properly lemmatize the text.
    """

    # create some inner functions for simplicity
    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']
    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']
    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']

    # check each tag against possible categories
    if is_noun(tag):
        return wn.NOUN
    elif is_verb(tag):
        return wn.VERB
    elif is_adverb(tag):
        return wn.ADV
    elif is_adjective(tag):
        return wn.ADJ
    else:
        return wn.NOUN

def Lemmatize(tokenlist):
    lemmatizer = WordNetLemmatizer()
    defaultPos = nltk.pos_tag(tokenlist) # get the POS tags from NLTK default tagger
    words_lemma = []
    for item in defaultPos:
        words_lemma.append(lemmatizer.lemmatize(item[0],pos_to_wn(item[1]))) # need to convert POS tags to a format (NOUN, VERB, ADV, ADJ) that wordnet uses to lemmatize
    return words_lemma

def ApplyPOSTagging(df,
                    filename,
                    add_stanford_tags=False,
                    stanford_pos_path=None,
                    stanford_language_path=None):

    """
    Given a dataframe of conversation turns, return a new
    dataframe with part-of-speech tagging. Add filename
    (given as string) as a new column in returned dataframe.

    By default, return only tags from the NLTK default POS
    tagger. Optionally, also return Stanford POS tagger
    results by setting `add_stanford_tags=True`.

    If Stanford POS tagging is desired, specify the
    location of the Stanford POS tagger with the
    `stanford_pos_path` argument. Also note that the
    default language model for the Stanford tagger is
    English (english-left3words-distsim.tagger). To change
    language model, specify the location with the
    `stanford_language_path` argument.

    """

    # if desired, import Stanford tagger
    if add_stanford_tags:
        if stanford_pos_path is None or stanford_language_path is None:
            raise ValueError('Error! Specify path to Stanford POS tagger and language model using the `stanford_pos_path` and `stanford_language_path` arguments')
        else:
            stanford_tagger = StanfordPOSTagger(stanford_pos_path + stanford_language_path,
                                                stanford_pos_path + 'stanford-postagger.jar')

    # add new columns to dataframe
    df['tagged_token'] = df['token'].apply(nltk.pos_tag)
    df['tagged_lemma'] = df['lemma'].apply(nltk.pos_tag)

    # if desired, also tag with Stanford tagger
    if add_stanford_tags:
        df['tagged_stan_token'] = df['token'].apply(stanford_tagger.tag)
        df['tagged_stan_lemma'] = df['lemma'].apply(stanford_tagger.tag)

    df['file'] = filename

    # return finished dataframe
    return df

def prepare_transcripts(input_files,
              output_file_directory,
              training_dictionary = None,
              minwords=2,
              use_filler_list=None,
              filler_regex_and_list=False,
              add_stanford_tags=False,
              stanford_pos_path=None,
              stanford_language_path=None,
              input_as_directory=True,
              save_concatenated_dataframe=True):

    """
    Given individual .txt files of conversations,
    return a completely prepared dataframe of transcribed
    conversations for later ALIGN analysis, including: text
    cleaning, merging adjacent turns, spell-checking,
    tokenization, lemmatization, and part-of-speech tagging.
    The output serve as the input for later ALIGN
    analysis.

    By default, use the Project Gutenberg corpus to create
    spell-checker (http://www.gutenberg.org). If desired,
    a different file may be used to train the spell-checker
    by setting `training_dictionary` to a path to the desired file.

    By default, set a minimum number of words in a turn to
    2. If desired, this may be chaged by changing the
    `minwords` file.

    By default, remove common fillers through regex.
    If desired, remove other words by passing a list
    of literal strings to `use_filler_list` argument,
    and if both regex and list of additional literal
    strings are to be used, update `filler_regex_and_list=True`.

    By default, return only the NLTK default
    POS tagger values. Optionally, also return Stanford POS
    tagger values with `add_stanford_tags=True`.

    If Stanford POS tagging is desired, specify the
    location of the Stanford POS tagger with the
    `stanford_pos_path` argument.

    By default, accept `input_files` as a directory
    that includes `.txt` files of each individual
    conversation. If desired, provide individual files
    as a list of literal paths to the `input_files`
    argument and set `input_as_directory=False`.

    By default, produce a single concatenated dataframe
    of all processed conversations in the output directory.
    If desired, suppress concatenated dataframe with
    `save_concatenated_dataframe=False`.
    """

    # create an internal function to train the model
    def train(features):
        model = defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    # if no training dictionary is specified, use the Gutenberg corpus
    if training_dictionary is None:

        # first, get the name of the package directory
        module_path = os.path.dirname(os.path.abspath(__file__))

        # then construct the path to the text file
        training_dictionary = os.path.join(module_path, 'data/gutenberg.txt')

    # train our spell-checking model
    nwords = train(re.findall('[a-z]+', (file(training_dictionary).read().lower())))

    # grab the appropriate files
    if not input_as_directory:
        file_list = glob.glob(input_files)
    else:
        file_list = glob.glob(input_files+"/*.txt")

    # cycle through all files
    main = pd.DataFrame()
    for fileName in file_list:

        # let us know which file we're processing
        dataframe = pd.read_csv(fileName, sep='\t',encoding='utf-8')
        print "Processing: "+fileName

        # clean up, merge, spellcheck, tokenize, lemmatize, and POS-tag
        dataframe = InitialCleanup(dataframe,
                                   minwords=minwords,
                                   use_filler_list=use_filler_list,
                                   filler_regex_and_list=filler_regex_and_list)
        dataframe = AdjacentMerge(dataframe)

        # tokenize and lemmatize
        dataframe['token'] = dataframe['content'].apply(Tokenize,
                                     args=(nwords,))
        dataframe['lemma'] = dataframe['token'].apply(Lemmatize)

        # apply part-of-speech tagging
        dataframe = ApplyPOSTagging(dataframe,
                                    filename=os.path.basename(fileName),
                                    add_stanford_tags=add_stanford_tags,
                                    stanford_pos_path=stanford_pos_path,
                                    stanford_language_path=stanford_language_path)

        # export the conversation's dataframe as a CSV
        conversation_file = os.path.join(output_file_directory,os.path.basename(fileName))
        dataframe.to_csv(conversation_file, encoding='utf-8',index=False,sep='\t')
        main = main.append(dataframe)

    # save the concatenated dataframe
    if save_concatenated_dataframe:
        concatenated_file = os.path.join(output_file_directory,'../align_concatenated_dataframe.txt')
        main.to_csv(concatenated_file,
                    encoding='utf-8',index=False, sep='\t')

    # return the dataframe
    return main
