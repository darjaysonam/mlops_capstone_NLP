import os
import sys

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.nlp.inference import ModelService
from .models import Prediction

model_service = ModelService()


def login_view(request):

    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "login.html")


@login_required
def dashboard(request):

    predictions = None

    if request.method == "POST":
        text = request.POST.get("text")

        if text:
            predictions = model_service.predict(text)

            Prediction.objects.create(
                user=request.user, input_text=text, result_json=predictions
            )

    history = Prediction.objects.filter(user=request.user).order_by("-created_at")

    return render(
        request, "dashboard.html", {"predictions": predictions, "history": history}
    )


@login_required
def logout_view(request):
    logout(request)
    return redirect("login")
