from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chest_pain = float(request.form['chest_pain'])
        resting_bp = float(request.form['resting_bp'])
        cholesterol = float(request.form['cholesterol'])
        fasting_blood_sugar = float(request.form['fasting_blood_sugar'])
        resting_ecg = float(request.form['resting_ecg'])
        max_heart_rate = float(request.form['max_heart_rate'])
        exercise_angina = float(request.form['exercise_angina'])
        st_depression = float(request.form['st_depression'])
        slope = float(request.form['slope'])
        num_major_vessels = float(request.form['num_major_vessels'])
        thalassemia = float(request.form['thalassemia'])

        # Prepare input data for prediction
        input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                                fasting_blood_sugar, resting_ecg, max_heart_rate,
                                exercise_angina, st_depression, slope,
                                num_major_vessels, thalassemia]])

        # Make prediction
        prediction = model.predict(input_data)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        
        return render_template('predict.html', result=result)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
