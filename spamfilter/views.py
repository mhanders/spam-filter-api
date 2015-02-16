import json
import sys
from django.http import HttpResponse, HttpResponseServerError
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from django import forms
from models import Distribution
from naivebayes import util, naivebayes

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        for thing in request.FILES:
            file = request.FILES[thing]
            for line in file:
                print >>sys.stderr, line.strip()
    c = util.Counter()
    print >>sys.stderr, c
    return HttpResponse('Done')

@csrf_exempt
def runbayes(request):
    distributions = Distribution.objects.all()
    # d = None
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

    (logProbabilities, logPrior, defaultProbabilities) = (json.loads(d.logProbabilities, encoding='latin-1'), \
        json.loads(d.logPrior), json.loads(d.defaultProbabilities))

    out = {}
    for (fileName, file) in request.FILES.items():
        out[fileName] = util.classifyMessage(file, logProbabilities, logPrior, defaultProbabilities)
    return HttpResponse(json.dumps(out))
    # return HttpResponse('A - OK')

def home(request):
    return HttpResponse("hey dude, welcome!")
