"""
Low Latency Edge AI Health Monitoring
Flask + ML + MongoDB Atlas Integration
"""
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import uuid

app = Flask(__name__)

# Load Trained ML Model
model = joblib.load("model.pkl")

# MongoDB Atlas Configuration
MONGO_URI = "mongodb+srv://HealthData:HealthData@healthdata.qwxicww.mongodb.net/?appName=HealthData"
client = MongoClient(MONGO_URI) 
db = client["health_ai"]
collection = db["predictions"]

# Prediction Helper
def predict_health(values):
    """
    Takes a list of vitals:
    [heart_rate, body_temp, oxygen, sys_bp, dia_bp, glucose]
    Returns: Normal / Alert / Critical (string)
    """
    prediction = model.predict([values])[0]
    label_map = {0: "Normal", 1: "Alert", 2: "Critical"}
    return label_map[prediction]


# Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_values = {}

    if request.method == "POST":
        try:
            # Collect form inputs
            patient_name = request.form["patient_name"]
            heart_rate = float(request.form["heart_rate"])
            body_temperature = float(request.form["body_temperature"])
            oxygen_level = float(request.form["oxygen_level"])
            bp_systolic = float(request.form["bp_systolic"])
            bp_diastolic = float(request.form["bp_diastolic"])
            glucose_level = float(request.form["glucose_level"])

            # Save input for returning to UI
            input_values = {
                "patient_name": patient_name,
                "heart_rate": heart_rate,
                "body_temperature": body_temperature,
                "oxygen_level": oxygen_level,
                "bp_systolic": bp_systolic,
                "bp_diastolic": bp_diastolic,
                "glucose_level": glucose_level
            }

            # Prepare list for model
            values = [
                heart_rate, body_temperature, oxygen_level,
                bp_systolic, bp_diastolic, glucose_level
            ]

            # Predict result
            result = predict_health(values)

            # Auto-generate a unique patient ID
            patient_id = str(uuid.uuid4())[:8]

            # Store into MongoDB Atlas
            record = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "vitals": input_values,
                "prediction": result,
                "timestamp": datetime.now()
            }

            collection.insert_one(record)

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, input_values=input_values)

# History Page
@app.route("/history")
def history():
    records = list(collection.find().sort("timestamp", -1))
    return render_template("history.html", records=records)

# Delete Record
@app.route("/delete/<id>")
def delete(id):
    collection.delete_one({"patient_id": id})
    return redirect(url_for("history"))


# Run App
if __name__ == "__main__":
    app.run(debug=True)
