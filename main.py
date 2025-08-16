from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

app = Flask(__name__)

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

@app.route('/train', methods=['POST'])
def train_model():
    file = request.files['dataset']
    target_col = request.form['target']

    df = pd.read_csv(file)
    df = clean_dataset(df)

    if target_col not in df.columns:
        return jsonify({"error": "Target column not found!"})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    return jsonify({"accuracy": round(score * 100, 2)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data["features"]

        # Dummy simple model for now
        prediction = sum(features) / len(features)  
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)