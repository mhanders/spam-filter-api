from __future__ import division
import sys
import os.path
import numpy as np
import util

NUM_CATEGORIES = 2

def learn_distributions(file_lists_by_category):
    log_probs_by_category = []
    prior_by_category = []

    for file_list in file_lists_by_category:
        log_prob = util.get_log_probabilities(file_list)
        log_probs_by_category.append(log_prob)
        prior_by_category.append(len(file_list))

    log_prior_by_category = [np.log(p/sum(prior_by_category)) for p in prior_by_category]

    return (log_probs_by_category, log_prior_by_category)

def classify_message(file,
                     log_probabilities_by_category,
                     log_prior_by_category,
                     default_probabilities,
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
                log_prob = default_probabilities[i]
            test = (w in message_words)
            total += test*log_prob + (1-test)*np.log(1-np.exp(log_prob))
        log_likelihoods.append(total)
    posterior = np.array(log_likelihoods) + np.array(log_prior_by_category)
    winner = np.argmax(posterior)
    return names[winner]

def update_log_probabilities(log_probabilities, files, previous_num_files, current_num_files):
        old_normalizer = np.log(previous_num_files + NUM_CATEGORIES)
        new_normalizer = np.log(current_num_files + NUM_CATEGORIES)

        counts = util.get_counts_from_request_files([x[1] for x in files.items()]) #Make sure no issues with double opening

        for word in log_probabilities:
            log_probabilities[word] += old_normalizer
            if (counts[word] > 0):
                log_probabilities[word] = np.log(np.exp(log_probabilities[word]) + counts[word])
                del counts[word]
            log_probabilities[word] -= new_normalizer
        for word in counts: # As deleted all encountered, above, these are new words in the training set
            log_probabilities[word] = np.log(counts[word] + 1) - new_normalizer
        return log_probabilities
