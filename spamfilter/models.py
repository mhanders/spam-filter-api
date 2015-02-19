import sys
import json
from django.db import models
from collections import defaultdict
from naivebayes import util
from naivebayes import naivebayes

SPAM_FOLDER = "naivebayes/data/wellham"
HAM_FOLDER = "naivebayes/data/wellspam"

LOG_HALF = "-0.69314718055994529"
EMPTY_DICT = "{}"

class Distribution(models.Model):
    num_spam              = models.IntegerField(default=0)
    num_ham               = models.IntegerField(default=0)
    log_priors            = models.TextField(default='['+LOG_HALF+', '+LOG_HALF+']')
    log_probabilities     = models.TextField(default=EMPTY_DICT)
    default_probabilities = models.TextField(default='') # Stored as (default_prob_for_spam, default_prob_for_ham)

    def learn(self):

        print >>sys.stderr, "Training..."
        file_lists = []
        for folder in (SPAM_FOLDER, HAM_FOLDER):
            file_lists.append(util.get_files_in_folder(folder))

        (log_probabilities_by_category, log_priors_by_category) = naivebayes.learn_distributions(file_lists)

        (self.log_probabilities, self.log_priors) = \
            (json.dumps(log_probabilities_by_category,ensure_ascii=False),\
             json.dumps(log_priors_by_category))

        # Store default probabilities, as these are lost when we serialize the dicts
        self.default_probabilities = json.dumps([log_probabilities_by_category[0][-1], log_probabilities_by_category[1][-1]])

        self.num_spam = len(file_lists[0])
        self.num_ham = len(file_lists[1])

    def save(self, *args, **kwargs):
        self.log_probabilities = self.log_probabilities.decode('latin-1')
        super(Distribution, self).save(*args, **kwargs)
