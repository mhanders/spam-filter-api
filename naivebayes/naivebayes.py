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
    message_words = set(file.read().split())

    totals = [log_prior_by_category[0], log_prior_by_category[1]]

    all_words = set(log_probabilities_by_category[0].keys()).union(log_probabilities_by_category[1].keys())

    for i in xrange(NUM_CATEGORIES):
        for word in all_words:
            if word in log_probabilities_by_category[i]:
                log_prob = log_probabilities_by_category[i][word]
            else:
                log_prob = default_probabilities[i]
            in_message = word in message_words
            totals[i] += in_message*log_prob + (1-in_message)*np.log(1 - np.exp(log_prob))
    if totals[0] > totals[1] or totals[1] - totals[0] < 100:
        return names[0]
    return names[1]

def update_log_probabilities(log_probabilities, files, previous_num_files, current_num_files):
        old_normalizer = np.log(previous_num_files + NUM_CATEGORIES)
        new_normalizer = np.log(current_num_files + NUM_CATEGORIES)

        counts = util.get_counts_from_request_files(files)
        for word in log_probabilities:
            log_probabilities[word] += old_normalizer
            if (counts[word] > 0):
                log_probabilities[word] = np.log(np.exp(log_probabilities[word]) + counts[word])
                del counts[word]
            log_probabilities[word] -= new_normalizer
        for word in counts: # As deleted all encountered, above, these are new words in the training set
            log_probabilities[word] = np.log(counts[word] + 1) - new_normalizer
        return log_probabilities
