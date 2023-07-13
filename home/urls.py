from django.contrib import admin
from django.urls import path, include
from home import views
from pycode.oly import fetch_medal_tally


urlpatterns = [
    path('', views.home, name='home'),
    path('fetch_medal/', views.fetch_medal, name='fetch_medal'),
    path('index/', views.index, name='index'),
    path('contact/', views.contact, name='contact'),
]