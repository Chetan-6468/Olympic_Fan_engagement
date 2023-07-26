import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

df = pd.read_csv(r"dataframe/shootouts.csv")
df.drop(['date'], axis=1, inplace=True)
df['result'] = np.where(df['home_team'] == df['winner'], 1, 0)
x = df.iloc[:, :-2]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf = ColumnTransformer([('trf', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'), ['home_team', 'away_team'])], remainder='passthrough')

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=[('step1', trf), ('steps2', LogisticRegression(solver='liblinear'))])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


@csrf_exempt
def football_prediction(request):
    if request.method == 'POST':
        data = request.POST
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        tournament = data.get('tournament')
        city = data.get('city')

        input_df = pd.DataFrame({'home_team': [home_team], 'away_team': [away_team]})
        y_pred_proba = pipe.predict_proba(input_df)

        # Extracting the probability of the positive class (home team winning) for the specific example
        predicted_probability_home = y_pred_proba[0, 1]
        predicted_probability_away = 1 - predicted_probability_home

        # Converting the probability to percentage and rounding to two decimal places
        predicted_percentage_home = round(predicted_probability_home * 100, 2)
        predicted_percentage_away = round(predicted_probability_away * 100, 2)

        return JsonResponse({
            'home_team_prediction': predicted_percentage_home,
            'away_team_prediction': predicted_percentage_away,
        })

    return JsonResponse({'error': 'Invalid request method.'})
