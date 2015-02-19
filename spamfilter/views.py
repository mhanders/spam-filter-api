import json
import sys
import numpy as np
from django.http import HttpResponse, HttpResponseServerError, HttpResponseBadRequest
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from django import forms
from models import Distribution
from naivebayes import util, naivebayes

### Testing features
TEST = False
END_TEST_MESSAGE = "Correctly classified %d/%d spam and %d/%d ham."
NOT_TESTING_MODE = "Testing not enabled. /test/ only available if TEST is enabled."
TESTING_HAM = "naivebayes/data/wellTesting/testingHardHam"
TESTING_SPAM = "naivebayes/data/wellTesting/testingSpam"
###

## HttpResponse messages
SUCCESSFULLY_TRAINED = "Training successful. Thank you for your help."
NOT_POST_MESSAGE = "Bad Request -- Only POST requests allowed"

## For (spam, ham)
NUM_CATEGORIES = 2

cached_distribution = None

def get_distribution():
    global cached_distribution
    if cached_distribution == None:
        distributions = Distribution.objects.all()
        if len(distributions) == 0:
            print >>sys.stderr, "Found none, so creating a new distribution..."
            d = Distribution()
            d.learn()
            d.save()
            cached_distribution = d
        else:
            cached_distribution = distributions[0]
    return cached_distribution

@csrf_exempt
def run_bayes(request):
    if request.method != 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)
    try:
        d = get_distribution()
    except:
        print >>sys.stderr, "Could not find or generate a distribution."
        return HttpResponseServerError()

    (log_probabilities, log_priors, default_probabilities) = (json.loads(d.log_probabilities, encoding='latin-1'), \
        json.loads(d.log_priors), json.loads(d.default_probabilities))

    out = {}
    for (fileName, files) in request.FILES.lists():
        if (len(files) > 1): # Mutliple files with the same name
            out[fileName] = [naivebayes.classify_message(file, log_probabilities, log_priors, default_probabilities) for file in files]
        else:
            out[fileName] = [naivebayes.classify_message(files[0], log_probabilities, log_priors, default_probabilities)]
    return HttpResponse(json.dumps(out))

@csrf_exempt
def train_ham(request):
    if request.method != 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)
    try:
        d = get_distribution()
    except:
        print >>sys.stderr, "Could not find or generate a distribution."
        return HttpResponseServerError()

    # We need to know the total number of files we've trained over, which is about to change.
    (num_spam, previous_num_spam) = (d.num_spam, d.num_ham)
    uploaded_files = []
    for (file_name, files) in request.FILES.lists():
        uploaded_files += files
    current_num_ham  = len(uploaded_files) + previous_num_ham

    # With more ham files, we need to update the ham default probability
    default_probabilities    = json.loads(d.default_probabilities)
    default_probabilities[1] = -np.log(current_num_ham + NUM_CATEGORIES)

    (log_probabilities, log_priors) = (json.loads(d.log_probabilities, encoding='latin-1'), json.loads(d.log_priors))

    log_probabilities[1] = naivebayes.update_log_probabilities(log_probabilities[1], uploaded_files, previous_num_ham, current_num_ham)

    # Priors also need updating based off of new file numbers
    spam_prior = (num_spam + 1.0) / (num_spam + current_num_ham + 2.0)
    log_priors = [np.log(spam_prior), np.log(1-spam_prior)]

    (d.log_probabilities, d.log_priors, d.default_probabilities) = \
        (json.dumps(log_probabilities, encoding='latin-1'), json.dumps(log_priors), json.dumps(default_probabilities))
    d.save()
    return HttpResponse(SUCCESSFULLY_TRAINED)

@csrf_exempt
def train_spam(request):
    if request.method != 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)
    try:
        d = get_distribution()
    except:
        print >>sys.stderr, "Could not find or generate a distribution."
        return HttpResponseServerError()

    # We need to know the total number of files we've trained over, which is about to change.
    (num_ham, previous_num_spam) = (d.num_ham, d.num_spam)
    uploaded_files = []
    for (file_name, files) in request.FILES.lists():
        uploaded_files += files
    current_num_spam = len(uploaded_files) + previous_num_spam

    # With more spam files, we need to update the spam default probability
    default_probabilities    = json.loads(d.default_probabilities)
    default_probabilities[0] = -np.log(current_num_spam + NUM_CATEGORIES)

    (log_probabilities, log_priors) = (json.loads(d.log_probabilities, encoding='latin-1'), json.loads(d.log_priors))

    log_probabilities[0] = naivebayes.update_log_probabilities(log_probabilities[0], uploaded_files, previous_num_spam, current_num_spam)

    # Priors also need updating based off of new file numbers
    spam_prior = (current_num_spam + 1.0) / (current_num_spam + num_ham + 2.0)
    log_priors = [np.log(spam_prior), np.log(1-spam_prior)]

    (d.log_probabilities, d.log_priors, d.default_probabilities) = \
        (json.dumps(log_probabilities, encoding='latin-1'), json.dumps(log_priors), json.dumps(default_probabilities))
    d.save()
    return HttpResponse(SUCCESSFULLY_TRAINED)


### Only runnable when TEST = True.
### Here as convenience simply to give performance stats.
def test(request):
    if not TEST:
        return HttpResponseBadRequest(NOT_TESTING_MODE)
    try:
        d = get_distribution()
    except:
        print >>sys.stderr, "Could not find or generate a distribution."
        return HttpResponseServerError()

    (log_probabilities, log_priors, default_probabilities) = (json.loads(d.log_probabilities, encoding='latin-1'), \
        json.loads(d.log_priors), json.loads(d.default_probabilities))

    file_lists = (util.get_files_in_folder(TESTING_HAM), util.get_files_in_folder(TESTING_SPAM))
    total_ham, total_spam, classified_ham, classified_spam = 0,0,0,0

    for f in file_lists[0]:
        with open(f, 'r') as ham_file:
            if naivebayes.classify_message(ham_file, log_probabilities, log_priors, default_probabilities) == 'ham':
                classified_ham += 1
            total_ham += 1
            if total_ham == 100: # Only test 100 ham files
                break
    for f in file_lists[1]:
        with open(f, 'r') as spam_file:
            if naivebayes.classify_message(spam_file, log_probabilities, log_priors, default_probabilities) == 'spam':
                classified_spam += 1
            total_spam += 1
            if total_spam == 100: # Only test 100 spam files
                break
    return HttpResponse(END_TEST_MESSAGE % (classified_spam, total_spam, classified_ham, total_ham))
