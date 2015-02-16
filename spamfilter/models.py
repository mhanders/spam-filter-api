import sys
import json
from django.db import models
from collections import defaultdict
from naivebayes import util
from naivebayes import naivebayes

TESTING_FOLDER = "naivebayes/data/testing"
SPAM_FOLDER = "naivebayes/data/superspam"
HAM_FOLDER = "naivebayes/data/superham"

LOG_HALF = "-0.69314718055994529"
EMPTY_DICT = "{}"

class Distribution(models.Model):
    name = models.CharField(max_length=200)
    logProbabilities = models.TextField(default=EMPTY_DICT)
    logPrior = models.TextField(default='['+LOG_HALF+', '+LOG_HALF+']')
    defaultProbabilities = models.TextField(default='')

    def learn(self):
        testing_folder = TESTING_FOLDER
        (spam_folder, ham_folder) = (SPAM_FOLDER, HAM_FOLDER)

        ### Learn the distributions
        print >>sys.stderr, "Training..."
        file_lists = []
        for folder in (spam_folder, ham_folder):
            file_lists.append(util.get_files_in_folder(folder))

        (log_probabilities_by_category, log_priors_by_category) = naivebayes.learn_distributions(file_lists)

        #Get default probabilities stored, as these are lost when we serialize the dicts
        self.defaultProbabilities = json.dumps([log_probabilities_by_category[0][-1], log_probabilities_by_category[1][-1]])


        (self.logProbabilities, self.logPrior) = \
            (json.dumps(log_probabilities_by_category,ensure_ascii=False),\
             json.dumps(log_priors_by_category))

    def save(self, *args, **kwargs):
        self.logProbabilities = self.logProbabilities.decode('latin-1')
        super(Distribution, self).save(*args, **kwargs)
