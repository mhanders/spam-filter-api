import sys
import json
from django.db import models
from collections import defaultdict
from naivebayes import util
from naivebayes import naivebayes

SPAM_FOLDER = "naivebayes/data/wellspam"
HAM_FOLDER = "naivebayes/data/wellham"

LOG_HALF = "-0.69314718055994529"
EMPTY_DICT = "{}"

class Distribution(models.Model):
    log_probabilities = models.TextField(default=EMPTY_DICT)
    log_priors = models.TextField(default='['+LOG_HALF+', '+LOG_HALF+']')
    default_probabilities = models.TextField(default='')
    num_spam = models.IntegerField(default=0)
    num_ham = models.IntegerField(default=0)

    def learn(self):
        (spam_folder, ham_folder) = (SPAM_FOLDER, HAM_FOLDER)

        ### Learn the distributions
        print >>sys.stderr, "Training..."

        file_lists = []
        for folder in (spam_folder, ham_folder):
            file_lists.append(util.get_files_in_folder(folder))

        (log_probabilities_by_category, log_priors_by_category) = naivebayes.learn_distributions(file_lists)

        (self.log_probabilities, self.log_priors) = \
            (json.dumps(log_probabilities_by_category,ensure_ascii=False),\
             json.dumps(log_priors_by_category))

        #Get default probabilities stored, as these are lost when we serialize the dicts
        self.default_probabilities = json.dumps([log_probabilities_by_category[0][-1], log_probabilities_by_category[1][-1]])

        self.num_spam = len(file_lists[0])
        self.num_ham = len(file_lists[1])

    def save(self, *args, **kwargs):
        self.log_probabilities = self.log_probabilities.decode('latin-1')
        super(Distribution, self).save(*args, **kwargs)
