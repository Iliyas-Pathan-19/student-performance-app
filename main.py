from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

app = Flask(__name__)

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
DATA_FILE = "Student_Performance.csv"

# Create dataset if missing
if not os.path.exists(DATA_FILE):
    sample_data = {
        "StudyHours": [2, 5, 7, 8, 3],
        "SleepHours": [6, 7, 8, 5, 6],
        "PreviousMarks": [50, 60, 70, 80, 55],
        "PapersPrepared": [1, 3, 4, 5, 2],
        "Age": [15, 16, 17, 16, 15],
        "Sex": ["M", "F", "M", "F", "M"],
        "Grade": [55, 65, 78, 82, 60],
    }
    pd.DataFrame(sample_data).to_csv(DATA_FILE, index=False)

df = pd.read_csv(DATA_FILE)

# Encode categorical
if "Sex" in df.columns:
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])  # M=1, F=0

X = df.drop("Grade", axis=1)
y = df["Grade"]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 2. Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([data])

        # Encode sex
        if "Sex" in features.columns:
            features["Sex"] = features["Sex"].map({"M": 1, "F": 0})

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        return jsonify({"predicted_grade": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# 3. Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
