<<<<<<< HEAD
import numpy as np
import pandas as pd
df=pd.read_csv(r"C:\Users\Lenovo\Desktop\fan engagement\olympic_analysis-web-app\gamedata.csv")
df.dropna()
def recommend(genre):
    temp_df=df[df['genre']==genre]
    return temp_df
recommend('Basketball')
=======
import pandas as pd

df = pd.read_csv("dataframe/gamedata.csv")
df.dropna(inplace=True)

def recommend(favorite_game):
    print("Favorite game received in recommend function:", favorite_game)

    favorite_game_lower = favorite_game.lower()
    temp_df = df[df["Title"].str.lower().str.contains(favorite_game_lower)]

    print("Recommendation DataFrame:")
    print(temp_df)

    recommendations = temp_df.to_dict(orient="records")
    print("Recommendations:")
    print(recommendations)

    return recommendations
>>>>>>> a92db9f0e9eb27fc4bc2104a0c19575010feeb4b
