# gestures/urls.py
from django.urls import path
from . import views


urlpatterns = [
    #path('', views.detect_gesture, name='detect_gesture'),
    path('', views.index, name='index'),
    path('start/', views.start, name='start'),
    path('stop/', views.stop, name='stop'),
]