from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# -----------------------------
# 1. Load and clean dataset
# -----------------------------
DATA_FILE = "Student_Performance.csv"

if not os.path.exists(DATA_FILE):
    # create a sample dataset if not exists
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

# Encode categorical columns
if "Sex" in df.columns:
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

X = df.drop("Grade", axis=1)
y = df["Grade"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
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

        prediction = model.predict(features)[0]
        return jsonify({"predicted_grade": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# 3. Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
