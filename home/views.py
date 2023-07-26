# views.py

from django.shortcuts import render,HttpResponse
import pandas as pd
from pycode.oly import fetch_medal_tally
from home.forms import inputform
from django.http import JsonResponse

def home(request):
    return render(request, 'front.html')

def index(request):
    return render(request, "index.html") 

def contact(request):
    return render(request, 'contact.html')

def about(request):
    return render(request, 'about.html')

def feature(request):
    return render(request, 'feature.html')

def stories(request):
    return render(request, 'stories.html')
import pycode.game
def game_recommendation(request):
    return render(request, 'game_recommendation.html')

def game_prediction(request):
    return render(request, 'game_prediction.html')

def football_game(request):
    return render(request, 'football_game.html')

def empty(request):
    return render(request, 'empty.html')

def image_classification(request):
    return render(request, 'image_classification.html')

from django.shortcuts import render


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




# -----------country wise analysis---------------



from pycode.oly import yearwise_medal, country_medal_heatmap, most_successful_athletecountry
import json
from django.http import JsonResponse


df = pd.read_csv(r'dataframe/athlete_events.csv')
region_df = pd.read_csv(r'dataframe/noc_regions.csv')

df = df[df['Season'] == 'Summer']
df = df.merge(region_df, on='NOC', how='left')
df.drop_duplicates(inplace=True)
df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)

from pycode.oly import fetch_medal_tally, yearwise_medal, country_medal_heatmap, most_successful_athletecountry

'''def country_analysis(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received data:", data)

            year = data.get('year')
            country = data.get('country')
            print("Year:", year)
            print("Country:", country)

            # Process the result to get specific data for country-wise analysis
            yearwise_medal_data = yearwise_medal(df, country)
            sportwise_medal_data = country_medal_heatmap(df, country)
            most_successful_athletes = most_successful_athletecountry(df, country)

            # Create a dictionary with the required data
            result_data = {
                'yearwiseMedal': yearwise_medal_data,
                'sportwiseMedal': sportwise_medal_data,
                'mostSuccessfulAthletes': most_successful_athletes.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
            }

            print("Result Data:", result_data)

            # Return the JSON response
            return JsonResponse(result_data)

        except json.JSONDecodeError as e:
            # Handle JSONDecodeError
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return render(request, 'country_analysis.html')
