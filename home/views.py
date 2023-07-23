# views.py

from django.shortcuts import render
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

def game_recommendation(request):
    return render(request, 'game_recommendation.html')

def game_prediction(request):
    return render(request, 'game_prediction.html')

def football_game(request):
    return render(request, 'football_game.html')

def empty(request):
    return render(request, 'empty.html')



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




# -----------country wise analysis---------------



df = pd.read_csv(r'dataframe/athlete_events.csv')
region_df = pd.read_csv(r'dataframe/noc_regions.csv')

df = df[df['Season'] == 'Summer']
df = df.merge(region_df, on='NOC', how='left')
df.drop_duplicates(inplace=True)
df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)

from pycode.oly import fetch_medal_tally, yearwise_medal, country_medal_heatmap, most_successful_athletecountry

def country_analysis(request):
    if request.method == 'POST':
        year = request.POST['year']
        country = request.POST['country']
       # result = fetch_medal_tally(year, country)   Call the function and store the result

        # Process the result to get specific data for country-wise analysis
        yearwise_medal_data = yearwise_medal(df, country)
        sportwise_medal_data = country_medal_heatmap(df, country)
        most_successful_athletes = most_successful_athletecountry(df, country)

        # Render the country_analysis.html template and pass the data as a context variable
        return render(request, 'country_analysis.html', {
            'yearwiseMedalData': yearwise_medal_data,
            'sportwiseMedalData': sportwise_medal_data,
            'mostSuccessfulAthletes': most_successful_athletes
        })

    return render(request, 'country_analysis.html')


# -----------------------game recommenation---------------

# views.py

from django.http import JsonResponse
from pycode.game import recommend
import pandas as pd
from django.views.decorators.csrf import csrf_exempt

df = pd.read_csv("dataframe/gamedata.csv")
df.dropna(inplace=True)

@csrf_exempt
def recommend_games(request):
    if request.method == "POST":
        favorite_game = str(request.POST.get("favorite-game", "")).strip()  # Convert to string and remove leading/trailing whitespaces
        print("Favorite game received from frontend:", favorite_game)

        recommendations = recommend(favorite_game)

        if not recommendations:
            print("No recommendations found for this game.")
            return JsonResponse([], safe=False)

        # Create a list of dictionaries with required keys for the frontend
        result = [{"Title": game["Title"], "genre": game["genre"]} for game in recommendations]
        return JsonResponse(result, safe=False, charset='utf-8')

    return JsonResponse({}, safe=False)
