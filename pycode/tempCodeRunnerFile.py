import pandas as pd
import numpy as np
df=pd.read_csv(r"dataframe/shootouts.csv")
df.drop(['date'],axis=1,inplace=True)
df['result']=np.where(df['home_team']==df['winner'],1,0)
x=df.iloc[:,:-2]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf=ColumnTransformer([('trf',OneHotEncoder(handle_unknown='ignore',sparse=False,drop='first'),['home_team','away_team'])],remainder='passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe=Pipeline(steps=[('step1',trf),('steps2',LogisticRegression(solver='liblinear'))])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
home_team=input("enter home team")
away_team=input('enter away team')
input_df=pd.DataFrame({'home_team':[home_team],'away_team':[away_team]})
pipe.predict_proba(input_df)