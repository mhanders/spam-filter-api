import json
import sys
from django.http import HttpResponse
from django.http import HttpResponseRedirect
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

def runbayes(request):
    distributions = Distribution.objects.all()
    if len(distributions) == 0:
        d = Distribution()
        d.learn()
    else:
        d = distributions[1]
    print >>sys.stderr, json.loads(d.logProbabilities, encoding='latin-1')
    return HttpResponse('A - OK')

def home(request):
    return HttpResponse("hey dude, welcome!")
