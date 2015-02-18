import json
import sys
import numpy as np
from django.http import HttpResponse, HttpResponseServerError, HttpResponseBadRequest
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from django import forms
from models import Distribution
from naivebayes import util, naivebayes

NOT_POST_MESSAGE = "Bad Request -- Only POST requests allowed"
NUM_CATEGORIES = 2

@csrf_exempt
def run_bayes(request):
    # Distribution.objects.all()[0].delete()
    if not request.method == 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)
    distributions = Distribution.objects.all()
    try:
        if len(distributions) == 0:
            print >>sys.stderr, "Found none, so creating a new distribution..."
            d = Distribution()
            d.learn()
            d.save()
        else:
            d = distributions[0]
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


# For additional training, these are the operations which must be performed:
# Update default_probabilities
# Update the log_probabilities by 1) Adding on correcting factor of log(nfiles - ncat)
#                                 2) Raising current prob to e, then adding on count
#                                 3) Adding back in new correcting factor of log(nfiles + newnumham + ncat)
# Update priors
# The methods below do the above.

@csrf_exempt
def train_ham(request):
    if not request.method == 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)
    d = Distribution.objects.all()[0]

    (num_spam, previous_num_spam) = (d.num_spam, d.num_ham)
    uploaded_files = []
    for (file_name, files) in request.FILES.lists():
        uploaded_files += files
    current_num_ham  = len(uploaded_files) + previous_num_ham #Handle some errors here

    default_probabilities    = json.loads(d.default_probabilities)
    default_probabilities[1] = -np.log(current_num_ham + NUM_CATEGORIES)

    (log_probabilities, log_priors) = (json.loads(d.log_probabilities, encoding='latin-1'), json.loads(d.log_priors))

    log_probabilities[1] = naivebayes.update_log_probabilities(log_probabilities[1], uploaded_files, previous_num_ham, current_num_ham)

    spam_prior = (num_spam + 1.0) / (num_spam + current_num_ham + 2.0)
    log_priors = [np.log(spam_prior), np.log(1-spam_prior)]

    (d.log_probabilities, d.log_priors, d.default_probabilities) = \
        (json.dumps(log_probabilities), json.dumps(log_priors), json.dumps(default_probabilities))
    d.save()
    return HttpResponse('Successfully trained!')

@csrf_exempt
def train_spam(request):
    if not request.method == 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)

    d = Distribution.objects.all()[0]

    (num_ham, previous_num_spam) = (d.num_ham, d.num_spam)
    uploaded_files = []
    for (file_name, files) in request.FILES.lists():
        uploaded_files += files
    current_num_spam = len(uploaded_files) + previous_num_spam

    default_probabilities    = json.loads(d.default_probabilities)
    default_probabilities[0] = -np.log(current_num_spam + NUM_CATEGORIES)

    (log_probabilities, log_priors) = (json.loads(d.log_probabilities, encoding='latin-1'), json.loads(d.log_priors))

    log_probabilities[0] = naivebayes.update_log_probabilities(log_probabilities[0], uploaded_files, previous_num_spam, current_num_spam)

    spam_prior = (current_num_spam + 1.0) / (current_num_spam + num_ham + 2.0)
    log_priors = [np.log(spam_prior), np.log(1-spam_prior)]

    (d.log_probabilities, d.log_priors, d.default_probabilities) = \
        (json.dumps(log_probabilities), json.dumps(log_priors), json.dumps(default_probabilities))
    d.save()
    return HttpResponse('Successfully trained!')
