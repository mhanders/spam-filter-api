import json
import sys
from django.http import HttpResponse, HttpResponseServerError
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from django import forms
from models import Distribution
from naivebayes import util, naivebayes

@csrf_exempt
def runbayes(request):
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

def test(request):
    d = Distribution.objects.all()[0]
    (log_probabilities, log_priors, default_probabilities) = (json.loads(d.log_probabilities, encoding='latin-1'), \
        json.loads(d.log_priors), json.loads(d.default_probabilities))
    file_lists = (util.get_files_in_folder('naivebayes/data/wellTesting/testingHardHam'), util.get_files_in_folder('naivebayes/data/wellTesting/testingspam'))
    totalHam, totalSpam, classifiedHam, classifiedSpam = 0,0,0,0
    for f in file_lists[0]:
        with open(f, 'r') as hamFile:
            print >>sys.stderr, f
            if naivebayes.classify_message(hamFile, log_probabilities, log_priors, default_probabilities) == 'ham':
                classifiedHam += 1
            totalHam += 1
            if totalHam == 100:
                break
    for f in file_lists[1]:
        with open(f, 'r') as spamFile:
            print >>sys.stderr, 'SPAM IS ' + f
            if naivebayes.classify_message(spamFile, log_probabilities, log_priors, default_probabilities) == 'spam':
                classifiedSpam += 1
            totalSpam += 1
            if totalSpam == 100:
                break
    return HttpResponse('Correctly classified ' + str(classifiedSpam) + '/' + str(totalSpam) \
        + ' spam and ' + str(classifiedHam) + '/' + str(totalHam))
