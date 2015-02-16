import sys
import os
import numpy as np

def get_words_in_file(filename):
    """ Returns a list of all words in the file at filename. """
    with open(filename, 'r') as f:
        # read() reads in a string from a file pointer, and split() splits a
        # string into words based on whitespace
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    """ Returns a list of files in folder (including the path to the file) """
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

class Counter(dict):
    """
    Like a dict, but returns 0 if the key isn't found.

    This is modeled after the collections.Counter class, which is
    only available in Python 2.7+. The full Counter class has many
    more features.
    """
    def __missing__(self, key):
        return 0

class DefaultDict(dict):
    """
    Like an ordinary dictionary, but returns the result of calling
    default_factory when the key is missing.

    For example, a counter (see above) could be implemented as either
    my_counter = Counter()
    my_counter = DefaultDict(lambda : 0)

    This is modeled after the collections.defaultdict class, which is
    only available in Python 2.7+.
    """

    def __init__(self, default_factory):
        """
        default_factory is a function that takes no arguments and
        returns the default value
        """
        self._default_factory = default_factory

    def __missing__(self, key):
        return self._default_factory()

def classifyMessage(file,
                     log_probabilities_by_category,
                     log_prior_by_category,
                     defaultProbabilities,
                     names = ['spam', 'ham']):
    """
    Uses Naive Bayes classification to classify the message in the given file.

    Inputs
    ------
    message_filename : name of the file containing the message to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    names : labels for each class (for this problem set, will always be just
            spam and ham).

    Output
    ------
    One of the labels in names.
    """

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
