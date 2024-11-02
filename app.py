# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form inputs
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        prediction = model.predict(final_features)
        output = 'Yes' if prediction[0] == 1 else 'No'
        return render_template('index.html', prediction_text=f'Heart Disease Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Invalid input.")

if __name__ == "__main__":
    app.run(debug=True)
