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


def health_pre(request):
    return render(request, 'health_pre.html')   


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



from pycode.oly import yearwise_medal, country_medal_heatmap, most_successful_athletecountry,most_successful
import json
from django.http import JsonResponse
import plotly.offline as pyo
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly as p
import plotly.graph_objects as go



df = pd.read_csv(r'dataframe/athlete_events.csv')
region_df = pd.read_csv(r'dataframe/noc_regions.csv')

df = df[df['Season'] == 'Summer']
df = df.merge(region_df, on='NOC', how='left')
df.drop_duplicates(inplace=True)
df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)

def yearwise(request):
    if request.method == 'POST':
        try:
            
            country = request.POST['country']
            

            # Process the result to get specific data for country-wise analysis
            yearwise_medal_data = yearwise_medal(df, country)
            # sportwise_medal_data = country_medal_heatmap(df, country)
            # most_successful_athletes = most_successful_athletecountry(df, country)

            # Create a dictionary with the required data
           
            yearwiseMedal=yearwise_medal_data
                # 'sportwiseMedal': sportwise_medal_data,
                # 'mostSuccessfulAthletes': most_successful_athletes.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
    

            #print("Result Data:", result_data)

            # Return the JSON response
            return render(request,'yearwise_data.html',{'yearwiseMedal':yearwiseMedal}
            )


        except json.JSONDecodeError as e:
            # Handle JSONDecodeError
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return render(request,'yearwise_data.html')
import base64
from io import BytesIO
def sportwise(request):
    if request.method == 'POST':
        try:
            
            country = request.POST['country']
            

            # Process the result to get specific data for country-wise analysis
            sportwise_medal_data = country_medal_heatmap(df, country)
            # sportwise_medal_data = country_medal_heatmap(df, country)
            # most_successful_athletes = most_successful_athletecountry(df, country)

            # Create a dictionary with the required data
            sns.heatmap(sportwise_medal_data.pivot_table(index='Sport',columns='Year',values='Medal',aggfunc='count').fillna(0).astype(int),annot=True)
            
            buffer=BytesIO()
            plt.savefig(buffer,format='png')
            plt.close()
            buffer.seek(0)
            image_base64=base64.b64encode(buffer.read().decode('utf-16'))
            
                # 'sportwiseMedal': sportwise
                # 'mostSuccessfulAthletes': most_successful_athletes.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
    

            #print("Result Data:", result_data)

            # Return the JSON response
            return render(request,'sportwise_data.html',{'image_base64':image_base64})
            


        except json.JSONDecodeError as e:
            # Handle JSONDecodeError
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return render(request,'sportwise_data.html')

def athletewise(request):
    if request.method == 'POST':
        try:
            
            country = request.POST['country']
            

            athlete_wise_Medal = most_successful_athletecountry(df, country)
            
            athletewiseMedal=athlete_wise_Medal.to_dict(orient='records')
            
                    
            return render(request,'athletewise_data.html',{'athletewiseMedal':athletewiseMedal})
            


        except json.JSONDecodeError as e:
            # Handle JSONDecodeError
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return render(request,'athletewise_data.html')

def top_statistics(request):
    edition=df['Year'].unique().shape[0]-1                         #no of edition
    host=df['City'].unique().shape[0]                              #no of cities
    sport=df['Sport'].unique().shape[0]                            #no of sports
    event=df['Event'].unique().shape[0]                            #no of events
    athletes=df['Name'].unique().shape[0]                          #no of athletes
    country=df['region'].unique().shape[0]
    return render(request,'top_stat.html',{'edition':edition,'host':host,'sport':sport,'event':event,'athletes':athletes,'country':country})

temp_df=df.dropna(subset='Medal')
temp_df['Name'].value_counts().reset_index().merge(temp_df, left_on='index', right_on='Name', how='left')[
        ['index', 'Name_x', 'Sport', 'region']].drop_duplicates('index')
def overall_mostsuccessfull_athlete(request):
    if request.method == 'POST':
        try:
            
            Sport = request.POST['Sport']
            

            overall_athlete_wise_Medal = most_successful(temp_df,Sport)
            
            overallathletewiseMedal=overall_athlete_wise_Medal.to_dict(orient='records')
            
                    
            return render(request,'overall_athlete.html',{'overallathletewiseMedal':overallathletewiseMedal})
            


        except json.JSONDecodeError as e:
            # Handle JSONDecodeError
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

    return render(request,'overall_athlete.html')
    
    
# -----------------------game recommenation---------------

from django.http import JsonResponse
from pycode.game import recommend
import pandas as pd
from django.views.decorators.csrf import csrf_exempt

df_2 = pd.read_csv("dataframe/gamedata.csv")
df_2.dropna(inplace=True)

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
df_1 = pd.read_csv('dataframe/shootouts.csv')
df_1.drop(['date'], axis=1, inplace=True)

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




# ----------------------health_prediction---------------


import numpy as np
import pandas as pd
from django.shortcuts import render
import pickle
from pycode.health_predict import minscaler, encoder, model, cols_scaled, encoded_cols, categorical_cols

def health_prediction_view(request):
    if request.method == 'POST':
        gender = request.POST.get('gender')
        age = request.POST.get('age')
        heart_rate = request.POST.get('heart_rate')
        temperature = request.POST.get('temperature')
        SpO2_saturation = request.POST.get('SpO2_saturation')
        bpm = request.POST.get('bpm')

        # Check if all fields are filled
        if not all([gender, age, heart_rate, temperature, SpO2_saturation, bpm]):
            return render(request, 'health_pre.html', {'error_message': 'Please fill all the details.'})

        # Convert data to float
        try:
            age = float(age)
            heart_rate = float(heart_rate)
            temperature = float(temperature)
            SpO2_saturation = float(SpO2_saturation)
            bpm = float(bpm)
        except ValueError:
            return render(request, 'health_pre.html', {'error_message': 'Invalid input. Please provide numeric values.'})

        # Preprocess the input data
        input_data = np.array([[gender, age, heart_rate, temperature, SpO2_saturation, bpm]])
        input_df = pd.DataFrame(input_data, columns=['gender', 'age', 'heart_rate', 'temperature', 'SpO2_saturation', 'bpm'])
        input_df['temperature'] = input_df['temperature'].astype(float).apply(lambda x: ((x - 32) * 5) / 9 if x > 80.0 else x)
        input_df[cols_scaled] = minscaler.transform(input_df[cols_scaled])
        input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
        input_df.drop(['gender'], axis=1, inplace=True)

        # Predict the health status
        model=pickle.load(open('model.pkl','rb'))
        prediction = model.predict(input_df)[0]
        if prediction==1:
            health_status='Athlete is fit'
        else:
            health_status='Athlete is not fit'

        return render(request, 'health_pre.html', {'health_status': health_status})

    return render(request, 'health_pre.html', {'health_status': None, 'error_message': None})
