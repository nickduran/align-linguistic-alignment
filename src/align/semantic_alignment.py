import os 
import glob
import pickle
import pandas as pd  
import numpy as np 
from tqdm import tqdm  
from sklearn.metrics.pairwise import cosine_similarity 

# For W2V specifically
import re 
import ast  
from collections import Counter  # Provides a way to count the frequency of elements in a collection
import gensim  # Library for unsupervised topic modeling and natural language processing
import gensim.downloader as api  # Downloads and loads pre-trained models and datasets
# from gensim.models import KeyedVectors  # Provides efficient word vector representation and storage

#For BERT specifically 
import torch
from transformers import BertTokenizer, BertModel



def pair_and_lag_columns(df: pd.DataFrame, columns_to_lag: list, suffix1: str = '1', suffix2: str = '2') -> pd.DataFrame:
    """
    Creates lagged pairs of specified columns, generating new columns with a 
    suffix of `suffix1` for the original content and `suffix2` for the lagged content. 
    Also adds a new column indicating the order of participants between successive rows.
    """
    for col in columns_to_lag:
        if col in df.columns:
            df[f'{col}{suffix1}'] = df[col]
            df[f'{col}{suffix2}'] = df[col].shift(-1)
    df['utter_order'] = df['participant'] + ' ' + df['participant'].shift(-1)
    return df

def calculate_cosine_similarity(df: pd.DataFrame, embedding_pairs: list) -> pd.DataFrame:
    """
    Computes cosine similarities between pairs of vectors in the specified columns 
    and adds the results as new columns in the DataFrame.
    """
    for col1, col2 in embedding_pairs:
        similarities = df.apply(
            lambda row: cosine_similarity(
                np.array(row[col1]).reshape(1, -1),
                np.array(row[col2]).reshape(1, -1)
            )[0][0] if row[col1] is not None and row[col2] is not None else None,
            axis=1
        )
        similarity_column_name = f"{col1}_{col2}_cosine_similarity"
        df[similarity_column_name] = similarities
    return df

# # retrieve the curent working directory where the script is being executed
# script_dir = os.getcwd()

# # create a directory called "gensim-data" (if doesn't already exist)
# local_cache_dir = os.path.join(script_dir, "gensim-data")
# os.makedirs(local_cache_dir, exist_ok=True)
# print(f"Local cache directory: {local_cache_dir}")

# # configure Gensim to use local_cache_dir as base directory for downloading and storing models
# api.BASE_DIR = local_cache_dir
# print(f"Gensim BASE_DIR set to: {api.BASE_DIR}")

# checks if specified models are already downloaded to cache directory, if not, download them
def download_and_cache_models(models, cache_dir):
    api.BASE_DIR = cache_dir
    for model_name in models:
        model_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(model_path):
            try:
                print(f"Downloading model: {model_name}")
                model = api.load(model_name)
                print(f"Downloaded and cached model: {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
        else:
            print(f"Model {model_name} already exists at: {model_path}")

# specifies the list of models to be cached locally, invoking the download_and_cache_models function 
# models_to_cache = ['word2vec-google-news-300', 'glove-twitter-200']
# download_and_cache_models(models_to_cache, local_cache_dir)

# attempts to load a model from specified file path
def load_model_if_not_exists(model_path, binary=True):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

# # checks if the w2v_google_model is already loaded in the global namespace. if not, attempts to load it from local cache directory.
# if 'w2v_google_model' not in globals():
#     w2v_google_model_path = os.path.join(local_cache_dir, 'word2vec-google-news-300', 'word2vec-google-news-300.gz')
#     w2v_google_model = load_model_if_not_exists(w2v_google_model_path, binary=True)
#     if w2v_google_model is not None:
#         print("Word2Vec Google News model loaded from local cache successfully.")
#     else:
#         print("Failed to load Word2Vec Google News model.")

# note: possible todo: is it more efficient to use gensim.downloader.load(model_name)?
# note, downloading model, it downloads properly, but also throws the exception warning for some reason. 
# TODO: instead of just loading google news model into global workspace, load in all within "models_to_cache"

def aggregate_conversations(folder_path: str) -> pd.DataFrame:
    """
    Aggregates multiple .txt files located in a specified folder 
    into a single pandas DataFrame. Each file is expected to be 
    tab-separated. 
    
    Returns a DataFrame containing the concatenated content of all 
    the .txt files
    """
    text_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
    concatenated_df = pd.DataFrame()

    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
    
    return concatenated_df

def build_filtered_vocab(data: pd.DataFrame, output_file_directory: str, high_sd_cutoff: float = 3, low_n_cutoff: int = 1):
    """
    Constructs a vocabulary from the ‘lemma’ column of the input DataFrame, 
    applying frequency-based filtering: ords occurring less frequently 
    than low_n_cutoff or more frequently than a certain standard deviation 
    above the mean (high_sd_cutoff) are filtered out. 
    
    Returns: Two lists: one with all vocabulary words and another with filtered words
    Outputs: The vocabulary frequencies to files
    """ 

    all_sentences = [re.sub(r'[^\w\s]+', '', str(row)).split() for row in data['lemma']]
    all_words = [word for sentence in all_sentences for word in sentence]

    frequency = Counter(all_words)

    frequency_filt = {word: freq for word, freq in frequency.items() if len(word) > 1 and freq > low_n_cutoff}
    
    if high_sd_cutoff is not None:
        mean_freq = np.mean(list(frequency_filt.values()))
        std_freq = np.std(list(frequency_filt.values()))
        cutoff_freq = mean_freq + (std_freq * high_sd_cutoff)
        filteredWords = {word: freq for word, freq in frequency_filt.items() if freq < cutoff_freq}
    else:
        filteredWords = frequency_filt
  
    vocabfreq_all = pd.DataFrame(frequency.items(), columns=["word", "count"]).sort_values(by='count', ascending=False)
    vocabfreq_filt = pd.DataFrame(filteredWords.items(), columns=["word", "count"]).sort_values(by='count', ascending=False)
  
    vocabfreq_all.to_csv(os.path.join(output_file_directory, 'vocab_unfilt_freqs.txt'), encoding='utf-8', index=False, sep='\t')
    vocabfreq_filt.to_csv(os.path.join(output_file_directory, 'vocab_filt_freqs.txt'), encoding='utf-8', index=False, sep='\t')
    
    return list(frequency.keys()), list(filteredWords.keys())

def is_list_like_column(series):
    """
    Checks if a pandas Series contains list-like strings (i.e., strings that 
    look like lists).
    """    

    try:
        return series.apply(lambda x: x.strip().startswith("[")).all()
    except AttributeError:
        return False

def convert_columns_to_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts any columns in a DataFrame that contain list-like strings into 
    actual Python lists using ast.literal_eval.
    """        

    columns_converted = []
    for col in df.columns:
        if is_list_like_column(df[col]):
            df[col] = df[col].apply(ast.literal_eval)
            columns_converted.append(col)
    return df, columns_converted

def get_sum_embeddings(token_list, model):
    """
    Calculates the sum of word embeddings for a list of tokens using a 
    pre-trained Word2Vec model.
    """ 

    if token_list is None:
        return None    
    embeddings = []
    for word in token_list:
        if word in model.key_to_index:  
            embeddings.append(model[word])    
    if embeddings:
        sum_embedding = np.sum(embeddings, axis=0)
        return sum_embedding
    else:
        return None  
    
def process_file_for_W2V(file_path, vocab_list: list,w2v_google_model):
    """
    Processes a file containing conversation data, filters tokens based on a provided vocabulary list,
    pairs and lags columns, computes word embeddings, and then calculates cosine similarities 
    between the embeddings.
    """
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    df, columns_converted = convert_columns_to_lists(df)

    # Filter tokens based on the vocabulary list
    columns_to_filter = ['lemma', 'token']
    for col in columns_to_filter:
        df[col] = df[col].apply(lambda token_list: [word for word in token_list if word in vocab_list])

    # Pair and lag the columns
    columns_to_lag = ['content', 'token', 'lemma']
    df = pair_and_lag_columns(df, columns_to_lag)

    # Compute embeddings
    for column in ["lemma", "token"]:
        df[f"{column}1_sum_embedding_W2V"] = df[f"{column}1"].apply(lambda tokens: get_sum_embeddings(tokens, w2v_google_model))
        df[f"{column}2_sum_embedding_W2V"] = df[f"{column}2"].apply(lambda tokens: get_sum_embeddings(tokens, w2v_google_model))

    # Calculate cosine similarities
    embedding_columns = [
        ("lemma1_sum_embedding_W2V", "lemma2_sum_embedding_W2V"),
        ("token1_sum_embedding_W2V", "token2_sum_embedding_W2V")
    ]
    df = calculate_cosine_similarity(df, embedding_columns)

    return df

def get_embedding_with_cache(text, embedding_cache, tokenizer,model):
    """
    Generates a BERT embedding for a given text while utilizing a cache to avoid redundant computations:
	•	  Checks if the embedding for the given text is already in the cache. If so, returns it.
	•	  If not cached, tokenizes the text, converts tokens to IDs, and feeds them to the BERT model to get the embedding.
	•	  The embedding is then averaged over all tokens and stored in the cache for future use.
    """ 
    
    if text is None:
      return None

    if text in embedding_cache:
      return embedding_cache[text]

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    if not token_ids:
        print(f"Warning: No valid tokens generated for text: '{text}'")
        return None  # Or handle with a default embedding
    token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
    token_ids = [token for token in token_ids if token is not None]

    #print(text, token_ids)
    # if not token_ids:
    
    #   print(f"Tokenization failed for text: '{text}'")
    #   return None
    
    input_ids = torch.tensor([token_ids])
    
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_states, dim=1).numpy()
    if embedding is None or embedding.size == 0:
        print(f"Error: Empty embedding for text: '{text}'")
        return None
    embedding_cache[text] = embedding

    return embedding

def process_file(file_path, embedding_cache, tokenizer, model, model_name):
    """
    Processes a single file to compute embeddings for pairs of utterances and 
    calculates the cosine similarity between these embeddings:
    • Reads the file into a DataFrame.
    • Pairs and lags the `content` column using `pair_and_lag_columns`.
    • Applies the `get_embedding_with_cache` function to each utterance pair 
      to compute embeddings.
    • Computes the cosine similarity between embeddings of successive utterances.
    """

    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    # Pair and lag the columns
    df = pair_and_lag_columns(df, columns_to_lag=['content'])

    # Compute embeddings for the lagged columns with model-specific suffixes
    for column in ["content1", "content2"]:
        df[f"{column}_embedding_{model_name}"] = df[column].apply(
            lambda text: get_embedding_with_cache(text, embedding_cache, tokenizer, model)
        )

    # Calculate cosine similarities between embeddings with the model name
    embedding_columns = [(f"content1_embedding_{model_name}", f"content2_embedding_{model_name}")]
    df = calculate_cosine_similarity(df, embedding_columns)

    return df


def semantic_alignment(
    folder_path,
    output_file_directory,
    BERT_embedding_cache_path,
    use_W2V=True,
    use_BERT=True
):
    # Prepare W2V processing if use_W2V is True
    concatenated_df1 = pd.DataFrame()
    if use_W2V:
        # Set up local cache for Gensim and load W2V model
        script_dir = os.getcwd()
        local_cache_dir = os.path.join(script_dir, "gensim-data")
        os.makedirs(local_cache_dir, exist_ok=True)
        print(f"Local cache directory: {local_cache_dir}")

        # Configure Gensim to use the local cache directory
        api.BASE_DIR = local_cache_dir
        models_to_cache = ['word2vec-google-news-300', 'glove-twitter-200']
        download_and_cache_models(models_to_cache, local_cache_dir)

        # Load Word2Vec Google News model
        if 'w2v_google_model' not in globals():
            w2v_google_model_path = os.path.join(local_cache_dir, 'word2vec-google-news-300', 'word2vec-google-news-300.gz')
            w2v_google_model = load_model_if_not_exists(w2v_google_model_path, binary=True)
            if w2v_google_model:
                print("Word2Vec Google News model loaded successfully.")
            else:
                print("Failed to load Word2Vec Google News model.")

        # Process files with Word2Vec
        os.makedirs(output_file_directory, exist_ok=True)
        text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        concatenated_text_files = aggregate_conversations(folder_path)
        vocab_all, vocab_filtered = build_filtered_vocab(concatenated_text_files, output_file_directory)
        for file_name in tqdm(text_files, desc="Processing files for W2V"):
            file_path = os.path.join(folder_path, file_name)
            df = process_file_for_W2V(file_path, vocab_filtered, w2v_google_model)
            concatenated_df1 = pd.concat([concatenated_df1, df], ignore_index=True)
    
    # Prepare BERT processing if use_BERT is True
    concatenated_df2 = pd.DataFrame()
    if use_BERT:

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Load or initialize BERT embedding cache
        try:
            with open(BERT_embedding_cache_path, "rb") as f:
                embedding_cache = pickle.load(f)
        except FileNotFoundError:
            embedding_cache = {}

        # Process files with BERT
        text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file_name in tqdm(text_files, desc="Processing files for BERT"):
            file_path = os.path.join(folder_path, file_name)
            df = process_file(file_path, embedding_cache, bert_tokenizer, bert_model,model_name="BERT")
            concatenated_df2 = pd.concat([concatenated_df2, df], ignore_index=True)

        # Save updated BERT embedding cache
        with open(BERT_embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)

    
    # Combine results based on which embeddings are enabled
    result_df = pd.DataFrame()
    if use_W2V:
        result_df = concatenated_df1
    if use_BERT:
        result_df = pd.concat([result_df, concatenated_df2], axis=1)
    
    
    # Check if result_df is still empty (no data processed)
    if result_df.empty:
        print("No embeddings were processed. Ensure at least one of use_W2V, use_BERT, or use_Llama is True.")
    else:
        # return result_df[["participant", 
        #                   "content",
        #                   "content1_embedding_BERT_content2_embedding_BERT_cosine_similarity"]
        return result_df