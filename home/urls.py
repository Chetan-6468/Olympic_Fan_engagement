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
    path('country_analysis/', views.country_analysis, name='country_analysis'),
    path('game_prediction/', views.game_prediction, name='game_prediction'),
    path('football_game/', views.football_game, name='football_game'),
    path('empty/', views.empty, name='empty'),
    path('image_classification/', views.image_classification, name='image_classification'),
    path('football_prediction/', views.football_prediction, name='football_prediction'),
    path('classify_image/', views.classify_image, name='classify_image'),
]
