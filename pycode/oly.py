import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

df = pd.read_csv(r'dataframe/athlete_events.csv')
region_df = pd.read_csv(r'dataframe/noc_regions.csv')

df = df[df['Season'] == 'Summer']
df = df.merge(region_df, on='NOC', how='left')
df.drop_duplicates(inplace=True)
df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])

# MEDAL_TALLY
def fetch_medal_tally(year, country):
    if year == 'overall' and country == 'overall':
        temp_df = medal_df
    elif year == 'overall' and country != 'overall':
        temp_df = medal_df[medal_df['region'] == country]
    elif year != 'overall' and country == 'overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    elif year != 'overall' and country != 'overall':
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

    x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()
    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    return x





#OVERALL_ANALYSIS
edition=df['Year'].unique().shape[0]-1                         #no of editions
host=df['City'].unique().shape[0]                              #no of cities
sport=df['Sport'].unique().shape[0]                            #no of sports
event=df['Event'].unique().shape[0]                            #no of events
athletes=df['Name'].unique().shape[0]                          #no of athletes
country=df['region'].unique().shape[0]                          #no of countries

#participating nations over the time
def nations_over(df):
    nations_over_time = df.drop_duplicates(['Year', 'region'])['Year'].value_counts().reset_index().sort_values('index')
    nations_over_time = nations_over_time.rename(columns={'index': 'Edition', 'Year': 'No of countries'})
    fig = px.line(nations_over_time, x='Edition', y='No of countries')
    fig.show()

#events over the time
def events_over(df):
    events_over_time=df.drop_duplicates(['Year','Event'])['Year'].value_counts().reset_index().sort_values('index')
    events_over_time=events_over_time.rename(columns={'index':'Edition','Year':'No of events'})
    fig=px.line(events_over_time,x='Edition',y='No of events')
    fig.show()

#athlete over the time
def athlete_over(df):
    athlete_over_time=df.drop_duplicates(['Year','Name'])['Year'].value_counts().reset_index().sort_values('index')
    athlete_over_time=athlete_over_time.rename(columns={'index':'Edition','Year':'No of athletes'})
    fig=px.line(athlete_over_time,x='Edition',y='No of athletes')
    fig.show()


#most successful athlete
temp_df=df.dropna(subset='Medal')
temp_df['Name'].value_counts().reset_index().merge(temp_df, left_on='index', right_on='Name', how='left')[
        ['index', 'Name_x', 'Sport', 'region']].drop_duplicates('index')
def most_successful(temp_df,sport):
    if sport!='overall':
        temp_df=temp_df[temp_df['Sport']==sport]
    x= temp_df['Name'].value_counts().reset_index().head(15).merge(df,left_on='index',right_on='Name',how='left')[['index','Name_x','Sport','region']].drop_duplicates('index')
    x.rename(columns={'index':'Name','Name_x':'Medals'},inplace=True)
    return x


#COUNTRY WISE ANALYSIS

import plotly.offline as pyo
import base64
from io import BytesIO
# Function to get year-wise medal data for a specific country
def yearwise_medal(df, country):
    temp_df = df.dropna(subset='Medal')
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)
    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()
    fig=px.line(final_df,x='Year',y='Medal')
    return fig.show()  # Convert DataFrame to a list of dictionaries

# Function to get sport-wise medal data for a specific country

# Function to get most successful athletes data for a specific country
def most_successful_athletecountry(df, country):
    temp_df=df.dropna(subset=['Medal'])
    temp_df=temp_df[temp_df['region']==country]
    
    x= temp_df['Name'].value_counts().reset_index().head(10).merge(df,left_on='index',right_on='Name',how='left')[['index','Name_x','Sport']].drop_duplicates('index')
    x.rename(columns={'index':'Name','Name_x':'Medals'},inplace=True)
    return x
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


#ATHLEYTE WISE ANALYSIS

# age dist
athlete_df = df.drop_duplicates(subset=['Name', 'region'])
def age_dist(athlete_df):
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'], show_hist=False, show_rug=False)
    fig.show()

# height dist
def height_dist(athlete_df):
    x1 = athlete_df['Height'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Height'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Height'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Height'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Height', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'], show_hist=False, show_rug=False)
    fig.show()

# weight dist
def weight_dist(athlete_df):
    x1 = athlete_df['Weight'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Weight'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Weight'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Weight'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Weight', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'], show_hist=False, show_rug=False)
    fig.show()


#winning gold wrt to age
famous_sports = ['Basketball', 'Judo', 'Football',  'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Polo', 'Ice Hockey']
def winning_gold(famous_sports):
    x=[]
    name=[]
    for sport in famous_sports:
        temp_df=athlete_df[athlete_df['Sport']==sport]
        x.append(temp_df[temp_df['Medal']=='Gold']['Age'].dropna())
        name.append(sport)
    fig=ff.create_distplot(x,name,show_hist=False,show_rug=False)
    fig.show()
