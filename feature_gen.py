# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm



def generate_vocabulary(stances, headlines, bodies):
	# Use this function to generate the vocabulary from the training data only
    
    # Convert the stances and bodies to a numpy array
    stances_array = pd.DataFrame(stances).as_matrix()
    bodies_array = pd.DataFrame(list(bodies.items())).as_matrix()
    headlines_array = pd.DataFrame(list(headlines.items())).as_matrix()
    
    # extract the bodies and headlines that are used in the train set only
    train_headlines_array = headlines_array[np.isin(headlines_array[:,0], stances_array[:,1]), :]
    train_bodies_array = bodies_array[np.isin(bodies_array[:,0], stances_array[:,0]), :]
    corpus = np.append(train_headlines_array[:,1], train_bodies_array[:,1])
    vocabulary = []
    
    for sentence in corpus:
        # Extract tokens
        tokens = re.findall(r"\w+(?:[-']\w+)*", sentence, flags=re.UNICODE)
        vocabulary.extend(tokens)
        
    vocabulary = list(set(vocabulary))
    
    return np.array(vocabulary)



def term_frequency_vectorizer(text, vocabulary):
    # This function takes a piece of text and converts it into a vectorized 
    # representation of term-frequencies (only for terms that are present in the vocabulary)

    words = re.findall(r"\w+(?:[-']\w+)*", text,  flags=re.UNICODE)
    # convert words to numpy array
    words = np.array(words)
    
    word_vec = np.zeros(len(vocabulary))
    unique_words = np.unique(words)
    
    for token in unique_words:
        word_vec[vocabulary == token] = words[words == token].size
                
    return word_vec.astype(np.float32)




def vectorize_dataset(headlines, bodies, vocabulary):
    # Generate a term-frequency vectors for all unique headlines and bodies
    # contained in the training/validation or testing set

    headlines_vectorised = {}
    bodies_vectorised = {}

    # Convert the stances and bodies to a numpy array
    headlines_array = pd.DataFrame(list(headlines.items())).as_matrix()
    bodies_array = pd.DataFrame(list(bodies.items())).as_matrix()
    
    # Vectorize the headlines
    for i, headline in tqdm(enumerate(headlines_array)):
        sentence_vec = term_frequency_vectorizer(headline[1], vocabulary)
        headlines_vectorised[headline[0]] = sentence_vec
        
    # Vectorize the bodies
    for i, body in tqdm(enumerate(bodies_array)):
        sentence_vec = term_frequency_vectorizer(body[1], vocabulary)
        bodies_vectorised[body[0]] = sentence_vec

    return headlines_vectorised, bodies_vectorised




def generate_tf_matrix(stances, headlines_vec, bodies_vec):
	# Use this function to generate a tf-matrix for the training set
	# this will be used to get idf weightings later

	# Convert the stances and bodies to a numpy array
    stances_array = pd.DataFrame(stances).as_matrix()
    headlines_array = pd.DataFrame(list(headlines_vec.items())).as_matrix()
    bodies_array = pd.DataFrame(list(bodies_vec.items())).as_matrix()
    
    # extract the bodies and headlines that are used in the train set only
    train_headlines_array = headlines_array[np.isin(headlines_array[:,0], stances_array[:,1]), :]
    train_bodies_array = bodies_array[np.isin(bodies_array[:,0], stances_array[:,0]), :]
    matrix_headlines = np.stack(train_headlines_array[:,1], axis=0)
    matrix_bodies = np.stack(train_bodies_array[:,1], axis=0)

    tf_matrix = np.concatenate((matrix_headlines, matrix_bodies))

    return tf_matrix




def inverse_doc_freq(tf_matrix):
    # Calculate the inverse document frequency for all terms in the vocabulary
    term_doc_counts = np.count_nonzero(tf_matrix, axis=0)
    # Implement idf with smoothing (smoothing technique is used by sklearn)
    idf = np.log(((tf_matrix.shape[0] + 1) / (term_doc_counts.astype(np.float32) + 1)) + 1)

    return idf.astype(np.float32)




def get_cosine_similarity(vec1, vec2):
    # This function claculates the cosine similaity between two vectors
    numerator = np.dot(vec1, vec2)
    denominator = (np.sqrt(np.square(vec1).sum()) * np.sqrt(np.square(vec2).sum()))

    # Avoid situation when denominator is 0
    if denominator == 0:
    	denominator = 1

    return numerator / float(denominator)




def feature_cosine_similarity(stances, headlines_vec, bodies_vec, tf_matrix):
    # Use this function to generate the cosine similarity between a set of 
    # stances and corresponding set of bodies
    
    feature = []

    # Convert the stances to a numpy array for easy indexing
    stances_array = pd.DataFrame(stances).as_matrix()

    # Get idf's of terms in vocabulary
    term_idfs = inverse_doc_freq(tf_matrix)

    # Iterate through the stances, get corresonding headline and body pair and calculate cosine similarity between them
    for stance in tqdm(stances_array):

        # Get corresponding body and headline tf-vectors
        body_vec = bodies_vec[stance[0]]
        headline_vec = headlines_vec[stance[1]]

        # Calculate tf-idf of headline and body
        headline_tf_idf = np.multiply(headline_vec, term_idfs)
        body_tf_idf = np.multiply(body_vec, term_idfs)

        # Calculate cosine similarity
        cos_sim = get_cosine_similarity(headline_tf_idf, body_tf_idf)
        #append the resutl to a list
        feature.append(cos_sim)

    return feature




def feature_kl_divergence(stances, headlines_vec, bodies_vec, tf_matrix, mu):
    # Use this function to generate the kl-divergence between the language models for a set of 
    # stances and the language models for their corresponding bodies
    
    feature = []
    
    # Convert the stances to a numpy array for easy indexing
    stances_array = pd.DataFrame(stances).as_matrix()
    
    # Calculte the probability of each word in the corpus
    # This information will be used for smooting 
    total_words_corpus = tf_matrix.sum()
    prob_corpus = tf_matrix.sum(axis=0) / float(total_words_corpus)
    
    # Iterate through the stances, get corresonding headline and body pair
    for stance in tqdm(stances_array):

        # Get corresponding body and headline tf-vectors
        body_vec = bodies_vec[stance[0]]
        headline_vec = headlines_vec[stance[1]]
        
        # Calculate p(w|headline) -- use dirichlet smoothing
        words_in_headline = headline_vec.sum()
        # Prevent situation when there are no words in the headline
        if words_in_headline == 0:
        	words_in_headline = 1
        prob_headline = headline_vec/float(words_in_headline)
        # Calculte lambda
        lambda_headline = words_in_headline/float(words_in_headline + mu)
        prob_headline_smooth = (lambda_headline*prob_headline) + ((1-lambda_headline)*prob_corpus)
        
        # Calculate p(w|body) -- use dirichlet smoothing
        words_in_body = body_vec.sum()
        # Prevent situation when there are no words in the body
        if words_in_body == 0:
        	words_in_body = 1
        prob_body = body_vec/words_in_body
        # Calculte lambda
        lambda_body = words_in_body/float(words_in_body + mu)
        prob_body_smooth = (lambda_body*prob_body) + ((1-lambda_body)*prob_corpus)
        
        # Calculate the KL-divergence
        kl_divergence = np.dot(prob_headline_smooth, (np.log10(prob_headline_smooth/prob_body_smooth)))
        feature.append(kl_divergence)
        
    return feature




def feature_jaccard_similarity(stances, headlines_vec, bodies_vec):
    # Use this function to generate the Jaccard similarity between a set of 
    # stances and their corresponding bodies
    
    feature = []
    
    # Convert the stances to a numpy array for easy indexing
    stances_array = pd.DataFrame(stances).as_matrix()
    
    # Iterate through the stances, get corresonding headline and body pair
    for stance in tqdm(stances_array):

        # Get corresponding body and headline tf-vectors
        body_vec = bodies_vec[stance[0]]
        headline_vec = headlines_vec[stance[1]]
        
        # Compute Jaccard similarity
        intersection = np.logical_and(headline_vec, body_vec).sum()
        union = np.logical_or(headline_vec, body_vec).sum()
        # Prevent situation when the union is empty
        if union == 0:
        	union = 1
        jaccard_sim = float(intersection)/union
        
        feature.append(jaccard_sim)
        
    return feature




def negative_feature(stances, headlines_vec, bodies_vec, vocabulary):
    # Use this function to count the number of refuting/negative tokens that are present in 
    # the headlines and bodies 
    
    feature = []
    
    # Get a list of 4782 negtive words 
    # source of words: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    with open("resources/neg_tokens.txt", "r") as f:
        neg_tokens = f.read()
    f.close()
    
    neg_tokens_vec = term_frequency_vectorizer(neg_tokens, vocabulary)
    
    # Convert the stances to a numpy array for easy indexing
    stances_array = pd.DataFrame(stances).as_matrix()
    
    # Iterate through the stances, get corresonding headline and body pair
    for stance in tqdm(stances_array):

        # Get corresponding body and headline tf-vectors
        body_vec = bodies_vec[stance[0]]
        headline_vec = headlines_vec[stance[1]]
        
        # Get the counts of negative tokens -- this can be done by getting the dot product between 
        # the vectors (Note: neg_tokens_vec is a binary vector)
        neg_tokens_headline = np.dot(neg_tokens_vec, headline_vec)
        neg_tokens_body = np.dot(neg_tokens_vec, body_vec)

        # normalise the vecotrs above
        headline_vec_count = float(headline_vec.sum())
        body_vec_count = float(body_vec.sum())
        
        # avoid divsion by 0
        if headline_vec_count == 0:
            headline_vec_count = 1
        if body_vec_count == 0:
            body_vec_count = 1

        normalised_neg_tokens_headline = (neg_tokens_headline/headline_vec_count)
        normalised_neg_tokens_body = (neg_tokens_body/body_vec_count)

        feature.append(np.absolute(normalised_neg_tokens_headline - normalised_neg_tokens_body))
        
    return feature




def generate_LSA_vectorizer(tf_matrix): 
    # generate a truncated-SVD vectorizer for performing LSA. 
    # Note: We restrict the vocabular to all the terms encountered in the training set only
    
    # We first create a tf-idf matrix, then perform truncated svd on this matrix
    term_idf = inverse_doc_freq(tf_matrix)
    tf_idf_matrix = np.multiply(tf_matrix, term_idf)
    
    # Generate a truncated SVD vectorizer based on the tfidf matrix, set number of components to 100
    svd_vectorizer = TruncatedSVD(n_components = 100).fit(tf_idf_matrix)
    
    return svd_vectorizer




def feature_LSA(stances, headlines_vec, bodies_vec, tf_matrix):
    # Use this function to generate the cosine similarity between a Latent semantic indexing 
    # representation of a set of stances and corresponding set of bodies

    # First generate LSI vectorizer based on tf-idf matrix
    lsi_vectorizer = generate_LSA_vectorizer(tf_matrix)

    # Get idf's of terms in vocabulary
    term_idfs = inverse_doc_freq(tf_matrix)
    
    feature = []
    
    # Convert the stances to a numpy array for easy indexing
    stances_array = pd.DataFrame(stances).as_matrix()
    
    # Iterate through the stances, get corresonding headline and body pair
    for stance in tqdm(stances_array):

        # Get corresponding body and headline tf-vectors
        body_vec = bodies_vec[stance[0]]
        headline_vec = headlines_vec[stance[1]]

        # Convert to tf-idf vector representation
        body_tfidf = np.multiply(body_vec, term_idfs)
        headline_tfidf = np.multiply(headline_vec, term_idfs)

        # Convert to LSI vector
        body_lsi = lsi_vectorizer.transform(body_tfidf.reshape(1,-1))
        headline_lsi = lsi_vectorizer.transform(headline_tfidf.reshape(1,-1))

        # Calculate the cosine similarity between the vectors
        cos_sim = get_cosine_similarity(headline_lsi[0,:], body_lsi[0,:])
        #append the results to a list
        feature.append(cos_sim)

    return feature