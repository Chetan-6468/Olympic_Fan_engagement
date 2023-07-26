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
df.dropna(inplace=True)

def country_analysis(request):
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

# -----------------------game recommenation---------------

from django.http import JsonResponse
from pycode.game import recommend
import pandas as pd
from django.views.decorators.csrf import csrf_exempt

df = pd.read_csv("dataframe/gamedata.csv")
df.dropna(inplace=True)

@csrf_exempt
def recommend_games(request):
    if request.method == "POST":
        favorite_game = str(request.POST.get("favorite-game", "")).strip()
        print("Favorite game received from frontend:", favorite_game)

        recommendations = recommend(favorite_game)

        if not recommendations:
            print("No recommendations found for this game.")
            return JsonResponse([], safe=False)

        # Create a list of game titles from the recommendations
        titles = [game["Title"] for game in recommendations]
        return JsonResponse(titles, safe=False, charset='utf-8')

    return JsonResponse({}, safe=False)




# -----------football prediction---------------------
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import random

# Load the dataset
df = pd.read_csv('dataframe/shootouts.csv')
df.drop(['date'], axis=1, inplace=True)

def football_prediction(request):
    if request.method == 'POST':
        # Get the form data
        home_team = request.POST.get('home_team')
        away_team = request.POST.get('away_team')

        # Perform the prediction using random values as probabilities
        predicted_probability_home = random.uniform(0, 1)
        predicted_probability_away = 1 - predicted_probability_home

        # Convert the probability to percentage and round to two decimal places
        predicted_percentage_home = round(predicted_probability_home * 100, 2)
        predicted_percentage_away = round(predicted_probability_away * 100, 2)

        # Return the predicted percentages as a JSON response
        return JsonResponse({
            'home_team_prediction': predicted_percentage_home,
            'away_team_prediction': predicted_percentage_away,
        })
    else:
        # Handle other request methods (GET, etc.) as needed
        return render(request, 'football_game.html')




#---------------------------image classification-----------------------------------

# classifier/views.py
import os
import pickle
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
import tensorflow_addons as tfa




# Load the trained model
model = load_model('path/to/trained/model.h5')

# Your code for loading the filenames and feature extraction
filenames = pickle.load(open('filenames.pkl', 'rb'))

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        
        # Save the uploaded image to a temporary location (optional)

        # Perform feature extraction (similar to your existing code)
        # Note: You can use the feature_extractor function defined in your image.py file
        uploaded_feature = feature_extractor('path/to/temporary/uploaded/image', model)

        # Find the closest matching celebrity from your dataset
        min_distance = float('inf')
        predicted_actor = None

        for filename in filenames:
            celebrity_feature = pickle.load(open('embedding.pkl', 'rb')) # Load features of celebrity images
            distance = np.linalg.norm(uploaded_feature - celebrity_feature)
            if distance < min_distance:
                min_distance = distance
                predicted_actor = os.path.basename(os.path.dirname(filename))
        
        # Remove the temporary uploaded image (optional)

        result_image_url = f"/static/sport_image/{predicted_actor}/{uploaded_image.name}"
        return JsonResponse({
            "predicted_actor": predicted_actor,
            "result_image_url": result_image_url
        })

    return render(request, 'image_classification.html')
