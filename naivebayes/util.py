import sys
import os
import numpy as np

def get_words_in_file(filename):
    with open(filename, 'r') as f:
        # read() reads in a string from a file pointer, and split() splits a
        # string into words based on whitespace
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

def classifyMessage(file,
                     log_probabilities_by_category,
                     log_prior_by_category,
                     defaultProbabilities,
                     names = ['spam', 'ham']):
                     
    # message_words = set(util.get_words_in_file(message_filename))
    message_words = set(file.read().split())
    N_categories  = len(log_probabilities_by_category)

    # get the union of all words encountered during training
    all_words = []
    for i in xrange(N_categories):
        all_words += log_probabilities_by_category[i].keys()
    all_words = list(set(all_words))

    log_likelihoods = []
    for i in xrange(N_categories):
        total = 0
        all_word_log_probs = log_probabilities_by_category[i]
        for w in all_words:
            if w in all_word_log_probs:
                log_prob = all_word_log_probs[w]
            else:
                log_prob = defaultProbabilities[i]
            test = (w in message_words)
            total += test*log_prob + (1-test)*np.log(1-np.exp(log_prob))
        log_likelihoods.append(total)
    posterior = np.array(log_likelihoods) + np.array(log_prior_by_category)
    winner = np.argmax(posterior)
    return names[winner]
