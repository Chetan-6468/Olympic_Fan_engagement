import pandas as pd

df_2 = pd.read_csv("dataframe/gamedata.csv")
df_2.dropna(inplace=True)

def recommend(favorite_game):
    print("Favorite game received in recommend function:", favorite_game)

    favorite_game_lower = favorite_game.lower()
    temp_df = df_2[df_2["Title"].str.lower().str.contains(favorite_game_lower)]
    temp_df['Title']
    print("Recommendation DataFrame:")
    print(temp_df)

    recommendations = temp_df.to_dict(orient="records")
    print("Recommendations:")
    print(recommendations)

    return recommendations
