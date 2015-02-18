import json
import sys
import numpy as np
from django.http import HttpResponse, HttpResponseServerError, HttpResponseBadRequest
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from django import forms
from models import Distribution
from naivebayes import util, naivebayes

NOT_POST_MESSAGE = "Only POST requests allowed"
NUM_CATEGORIES = 2

@csrf_exempt
def runbayes(request):
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
    for (fileName, file) in request.FILES.items():
        out[fileName] = naivebayes.classify_message(file, log_probabilities, log_priors, default_probabilities)
    return HttpResponse(json.dumps(out))

@csrf_exempt
def train_ham(request):
    if not request.method == 'POST':
        return HttpResponseBadRequest(NOT_POST_MESSAGE)

    # Update defaultProbabilities
    # Update the log_probabilities by 1) Adding on correcting factor of log(nfiles - ncat)
    #                                 2) Raising current prob to e, then adding on count
    #                                 3) Adding back in new correcting factor of log(nfiles + newnumham + ncat)
    #                                 4) Updating prior with
    #                                                   s = (len(file_lists_by_category[0]) + 1)/(len(file_lists_by_category[0]) + len(file_lists_by_category[1]) + 2)
    #                                                   log_prior_by_category = [np.log(s), np.log(1-s)]

    d                = Distribution.objects.all()[0]
    num_spam         = d.num_spam
    previous_num_ham = d.num_ham
    total_ham        = len(request.FILES.items()) + previous_num_ham #Handle some errors here

    default_probabilities    = json.loads(d.default_probabilities)
    default_probabilities[1] = -np.log(total_ham + NUM_CATEGORIES)

    counts = util.get_counts_from_request_files([x[1] for x in request.FILES.items()]) #Make sure no issues with double opening

    (log_probabilities, log_priors) = (json.loads(d.log_probabilities, encoding='latin-1'), json.loads(d.log_priors))
    ham_log_probabilities = log_probabilities[1]

    old_normalizer = np.log(previous_num_ham + NUM_CATEGORIES)
    new_normalizer = np.log(total_ham + NUM_CATEGORIES)

    for word in ham_log_probabilities:
        ham_log_probabilities[word] += old_normalizer
        if (counts[word] > 0):
            ham_log_probabilities[word] = np.log(np.exp(ham_log_probabilities[word]) + counts[word])
            del counts[word]
        ham_log_probabilities[word] -= new_normalizer
    for word in counts: # As deleted all encountered, above, these are new words in the training set
        ham_log_probabilities[word] = np.log(counts[word] + 1) - new_normalizer

    log_probabilities[1] = ham_log_probabilities

    spam_prior = (num_spam + 1.0) / (num_spam + total_ham + 2.0)
    log_priors = [np.log(spam_prior), np.log(1-spam_prior)]

    (d.log_probabilities, d.log_priors, d.default_probabilities) = \
        (json.dumps(log_probabilities), json.dumps(log_priors), json.dumps(default_probabilities))
    d.save()
    return HttpResponse('Successfully trained!')
