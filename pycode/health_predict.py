import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler


health_data = pd.read_csv('dataframe/health_data.csv')
health_data.head()

n_rows = health_data.shape[0]
n_cols = health_data.shape[1]
# print(f"The data has {n_rows} rows and {n_cols} columns")

health_data['gender'].isnull().sum()
health_data_not_missing = health_data.dropna(axis=0)
health_data_not_missing.isna().sum().sum()

# health_data_not_missing.info()
health_data_not_missing.describe()
health_data_not_missing[health_data_not_missing.Health_status == 1].count()


health_data_not_gender = health_data_not_missing.iloc[0:,2:]
health_data_not_gender.corr()

health_data_not_missing.temperature
temperature_series = health_data_not_missing.temperature
n = temperature_series.size
temperature_series.__class__
for i in range(n):
    if(temperature_series.iloc[i] > 80.0):
         Fahrenheit= temperature_series.iloc[i]
         Celsius = ((Fahrenheit-32)*5)/9
         temperature_series.iloc[i] = Celsius

temperature_series_df = pd.DataFrame(temperature_series)


input_cols = list(health_data_not_missing.columns[1:-1])

output_cols = 'Health_status'

input_df = health_data_not_missing[input_cols].copy()

output_df = health_data_not_missing[output_cols].copy()

cols_scaled = input_cols[1:]
# cols_scaled

scaler = StandardScaler()
scaler = StandardScaler().fit(input_df[cols_scaled])

minscaler = MaxAbsScaler()
minscaler = MaxAbsScaler().fit(input_df[cols_scaled])

input_df[cols_scaled] = minscaler.transform(input_df[cols_scaled])

categorical_cols = input_df.select_dtypes('object').columns.tolist()
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(input_df[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
# print(encoded_cols)

input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])

input_df.drop(['gender'] , axis=1 , inplace=True)

from sklearn.model_selection import train_test_split
input_train,input_test,train_target,test_target = train_test_split(input_df,output_df,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(solver='liblinear' , random_state=42)
model1.fit(input_train,train_target)

numeric_cols = input_train.select_dtypes(include=np.number).columns.tolist()[:-2]
pred = model1.predict(input_train)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(train_target,pred,normalize='true')

from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=20,max_depth=5)
model2.fit(input_train,train_target)

model2.score(input_train,train_target)

model2.score(input_test,test_target)


pickle.dump(model2,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))