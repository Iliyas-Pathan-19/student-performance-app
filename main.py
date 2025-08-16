from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "student_model.pkl"
model = None
features_list = []


# --- Helper: clean dataset ---
def clean_dataset(df):
    df = df.dropna(axis=1, how="all")
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


@app.route('/')
def home():
    return render_template("index.html")


# --- Train model from CSV ---
@app.route('/train', methods=['POST'])
def train_model():
    global model, features_list

    file = request.files['dataset']
    target_col = request.form['target']

    df = pd.read_csv(file)
    df = clean_dataset(df)

    if target_col not in df.columns:
        return jsonify({"error": "Target column not found!"})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    features_list = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))

    # save model
    joblib.dump(model, MODEL_PATH)

    return jsonify({
        "accuracy": round(score * 100, 2),
        "features": features_list
    })


# --- Manual Prediction (without CSV) ---
@app.route('/predict', methods=['POST'])
def predict():
    global model, features_list

    try:
        if model is None and os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)

        if model is None:
            return jsonify({"error": "Model not trained yet! Please upload CSV and train first."})

        data = request.json
        features = data["features"]  # dictionary of inputs

        # Make sure the input matches feature order
        input_values = [features.get(f, 0) for f in features_list]

        prediction = model.predict([input_values])[0]

        # Give advice based on study/sleep balance
        advice = []
        if "study_hours" in features and features["study_hours"] < 2:
            advice.append("Increase study hours 📚")
        if "sleep_hours" in features and features["sleep_hours"] < 6:
            advice.append("You need more sleep 😴")
        if "previous_marks" in features and features["previous_marks"] < 50:
            advice.append("Focus on revising weak topics 📖")
        if "papers_prepared" in features and features["papers_prepared"] < 2:
            advice.append("Try solving more practice papers 📝")

        return jsonify({
            "prediction": round(prediction, 2),
            "advice": advice if advice else ["Keep going, you are doing well! 🚀"]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)