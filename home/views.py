# views.py

from django.shortcuts import render
import pandas as pd
from pycode.oly import fetch_medal_tally
from home.forms import inputform


def home(request):
    return render(request, 'front.html')

def index(request):
    return render(request, "index.html") 

def contact(request):
    return render(request, 'contact.html')

from django.shortcuts import render

# def index(request):
#     return render(request, 'index.html', {'fetch_medal': fetch_medal_tally()})



import json
from django.http import JsonResponse  

def fetch_medal(request):
    if request.method == 'POST':
        year = request.POST['year']
        country = request.POST['country']
        result = fetch_medal_tally(year, country)  # Call the function and store the result
        
        # Convert DataFrame to a list of dictionaries
        result_json = result.to_dict(orient='records')
        
        return JsonResponse(result_json, safe=False)  # Return the converted result as a JSON response

    return JsonResponse({}, safe=False)  # Return an empty JSON response if the request is not a POST request

