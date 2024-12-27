# gestures/views.py
from django.http import JsonResponse
from django.shortcuts import render
from .gesture_recognition import start_recognition, stop_recognition





def index(request):
    return render(request, 'gestures/index.html')

def start(request):
    start_recognition()
    return JsonResponse({"status": "Gesture recognition started"})

def stop(request):
    stop_recognition()
    return JsonResponse({"status": "Gesture recognition stopped"})


