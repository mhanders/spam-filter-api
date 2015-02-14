import json
import sys
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
from django import forms

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        for thing in request.FILES:
            file = request.FILES[thing]
            for line in file:
                print >>sys.stderr, line.strip()
    return HttpResponse('done')

def home(request):
    # print request.raw_post_data
    print request.POST
    return HttpResponse("hey dude, welcome!")
