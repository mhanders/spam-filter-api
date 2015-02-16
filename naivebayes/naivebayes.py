from __future__ import division
import sys
import os.path
import numpy as np

import util

from collections import Counter, defaultdict

def get_counts(file_list):

    counts = Counter()
    for f in file_list:
        words = util.get_words_in_file(f)
        for w in set(words):
            counts[w] += 1
    return counts

def get_log_probabilities(file_list):
    counts = get_counts(file_list)
    N_files = len(file_list)
    N_categories = 2
    log_prob = defaultdict(lambda : -np.log(N_files + N_categories))
    for word in counts:
        log_prob[word] = np.log(counts[word] + 1) - np.log(N_files + N_categories)
        assert log_prob[word] < 0
    return log_prob


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
        log_prob = get_log_probabilities(file_list)
        log_probs_by_category.append(log_prob)
        prior_by_category.append(len(file_list))

    log_prior_by_category = [np.log(p/sum(prior_by_category)) for p in prior_by_category]

    return (log_probs_by_category, log_prior_by_category)
