import numpy as np
import pandas as pd
df=pd.read_csv(r"C:\Users\Lenovo\Desktop\fan engagement\olympic_analysis-web-app\gamedata.csv")
df.dropna()
def recommend(genre):
    temp_df=df[df['genre']==genre]
    return temp_df
recommend('Basketball')