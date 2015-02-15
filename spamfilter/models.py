import sys
import json
from django.db import models
from collections import defaultdict
from naivebayes import util
from naivebayes import naivebayes

TESTING_FOLDER = "naivebayes/data/testing"
SPAM_FOLDER = "naivebayes/data/spam"
HAM_FOLDER = "naivebayes/data/ham"

LOG_HALF = "-0.69314718055994529"
EMPTY_DICT = "{}"

class Distribution(models.Model):
    name = models.CharField(max_length=200)
    logProbabilities = models.TextField(default=EMPTY_DICT)
    logPrior = models.TextField(default="["+LOG_HALF+", "+LOG_HALF+"]")

    def learn(self):
        testing_folder = TESTING_FOLDER
        (spam_folder, ham_folder) = (SPAM_FOLDER, HAM_FOLDER)

        ### Learn the distributions
        print >>sys.stderr, "Training..."
        file_lists = []
        for folder in (spam_folder, ham_folder):
            file_lists.append(util.get_files_in_folder(folder))
        (log_probabilities_by_category, log_priors_by_category) = \
                naivebayes.learn_distributions(file_lists)

        # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
        # rows correspond to true label, columns correspond to guessed label
        performance_measures = np.zeros([2,2])

        (self.logProbabilities, self.logPrior) = (json.dumps(log_probabilities_by_category, ensure_ascii=False).decode('latin-1'), json.dumps(log_priors_by_category))

        # Used for measuring performance.
        # self.performanceMeasures = json.dumps(performance_measures)

    def save(self, *args, **kwargs):
        self.logProbabilities = self.logProbabilities.decode('latin-1')
        super(Distribution, self).save(*args, **kwargs)
