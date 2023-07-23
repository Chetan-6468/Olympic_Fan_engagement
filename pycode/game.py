# game.py

import pandas as pd

df = pd.read_csv("dataframe/gamedata.csv")
df.dropna(inplace=True)

def recommend(favorite_game):
    temp_df = df[df["genre"] == favorite_game]
    print("Filtered DataFrame:")
    print(temp_df)
    return temp_df.to_dict(orient="records")
