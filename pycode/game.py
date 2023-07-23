# game.py

import pandas as pd

df = pd.read_csv("dataframe/gamedata.csv")
df.dropna(inplace=True)

def recommend(favorite_game):
    print("Favorite game received in recommend function:", favorite_game)  # Add this line to check the value of favorite_game

    # Convert both the user input and the "Title" column to lowercase for case-insensitive filtering
    favorite_game_lower = favorite_game.lower()
    temp_df = df[df["Title"].str.lower().str.contains(favorite_game_lower)]
    print("Recommendation DataFrame:")
    print(temp_df)  # Add this line to check the filtered DataFrame

    recommendations = temp_df.to_dict(orient="records")
    print("Recommendations:")
    print(recommendations)  # Add this line to check the recommendations

    return recommendations
