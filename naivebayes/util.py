import sys
import os
import numpy as np
from collections import Counter, defaultdict

def get_files_in_folder(folder):
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

def get_counts(file_list):
    counts = Counter()
    for filename in file_list:
        with open(filename, 'r') as f:
            f = open(filename, 'r')
            words = f.read().split()
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
