from django.contrib import admin
from django.urls import path, include
from home import views
from pycode.oly import fetch_medal_tally


urlpatterns = [
    path('', views.home, name='home'),
    path('fetch_medal/', views.fetch_medal, name='fetch_medal'),
    path('index/', views.index, name='index'),
    path('contact/', views.contact, name='contact'),
    path('about/', views.about, name='about'),
    path('feature/', views.feature, name='feature'),
    path('stories/', views.stories, name='stories'),
    path('game_recommendation/', views.game_recommendation, name='game_recommendation'),
     path("recommend_games/", views.recommend_games, name="recommend_games"),

    path('yearwise/', views.yearwise, name='yearwise'),
    path('top_statistics/', views.top_statistics, name='top_statistics'),
    path('overall_mostsuccessfull_athlete/', views.overall_mostsuccessfull_athlete, name='overall_mostsuccessfull_athlete'),
    path('sportwise/', views.sportwise, name='sportwise'),
    path('athletewise/', views.athletewise, name='athletewise'),
    path('game_prediction/', views.game_prediction, name='game_prediction'),
    path('football_game/', views.football_game, name='football_game'),
    path('empty/', views.empty, name='empty'),
    
    path('football_prediction/', views.football_prediction, name='football_prediction'),
    
    path('health_pre/', views.health_pre, name='health_pre'),
    path('health_prediction_view/', views.health_prediction_view, name='health_prediction_view'),

]
