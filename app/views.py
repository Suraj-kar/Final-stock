import os
import sys
import numpy as np
import joblib
from django.shortcuts import render
from django.contrib import messages  # For success/error messages
from .models import StockLog

# Set up paths correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_models")

# Fix sys.path before importing Regression
sys.path.append(os.path.join(BASE_DIR, "model"))
from model.Regression import MultiLinearRegression, DataScaler  # Import properly

def get_available_companies():
    """Fetch the available company names from saved models."""
    files = os.listdir(MODEL_DIR)
    return [f.replace("_model.pkl", "") for f in files if f.endswith("_model.pkl")]

def home(request):
    data = {}
    company_names = get_available_companies()
    logs = StockLog.objects.all().order_by('-id')[:10]  # Fetch latest 10 logs

    if request.method == "POST":
        print("POST request received!")
        print("Received Data:", request.POST)

        try:
            company_name = request.POST.get("company")
            open_val = request.POST.get("open")
            high_val = request.POST.get("high")
            low_val = request.POST.get("low")
            vol_val = request.POST.get("vol")

            # Convert values to float & Auto-Calculate High & Low
            open_val = float(open_val) if open_val else None
            high_val = float(high_val) if high_val else (round(open_val * 1.1, 2) if open_val else None)
            low_val = float(low_val) if low_val else (round(open_val * 0.9, 2) if open_val else None)
            vol_val = float(vol_val) if vol_val else None

            print(f"Company: {company_name}, Open: {open_val}, High: {high_val}, Low: {low_val}, Vol: {vol_val}")

            if not company_name or None in [open_val, high_val, low_val, vol_val]:
                raise ValueError("Missing input values!")

            # Fix model paths
            scaler_path = os.path.join(MODEL_DIR, f"{company_name}_scaler.pkl")
            model_path = os.path.join(MODEL_DIR, f"{company_name}_model.pkl")

            if not os.path.exists(scaler_path) or not os.path.exists(model_path):
                messages.error(request, f"Model or scaler missing for {company_name}. Please try another company.")
                return render(request, "home.html", {"data": data, "companies": company_names, "logs": logs})

            # Load model and scaler properly
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path, mmap_mode=None)

            # Preprocess and predict
            X = np.array([[open_val, high_val, low_val, vol_val]])
            X_scaled = scaler.transform(X)
            result = model.predict(X_scaled)[0]

            percentage_change = ((result - open_val) / open_val) * 100
            difference = result - open_val

            print(f"Prediction: {result}, % Change: {percentage_change:.2f}, Difference: {difference:.2f}")

            # Save to database
            StockLog.objects.create(
                company=company_name,
                open_price=open_val,
                high_price=high_val,
                low_price=low_val,
                volume=vol_val,
                result=result,
                percentage_change=percentage_change,
                difference=difference,
            )
            print("Data saved to database!")

            # Pass data to template
            data = {
                "company": company_name,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "vol": vol_val,
                "result": f"{result:.2f}",
                "percentage_change": f"{percentage_change:.2f}",
                "difference": f"{difference:.2f}",
            }
            messages.success(request, "Stock data submitted successfully!")
        
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
            print("Error:", str(e))

    return render(request, "home.html", {"data": data, "companies": company_names, "logs": logs})

def stocklogs_view(request):
    datastocklog = StockLog.objects.all().order_by("-id")
    return render(request, "stockhistory.html", {"stocklog": datastocklog})