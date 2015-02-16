from __future__ import division
import sys
import os.path
import numpy as np

import util

def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files,
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
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
