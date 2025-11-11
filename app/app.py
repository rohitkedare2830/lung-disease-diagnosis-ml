from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values in correct order (Age, Height, Smoke, Gender, Caesarean)
        age = float(request.form['Age'])
        height = float(request.form['Height'])
        smoke = float(request.form['Smoke'])
        gender = float(request.form['Gender'])
        caesarean = float(request.form['Caesarean'])

        # Arrange features as numpy array
        features = np.array([[age, height, smoke, gender, caesarean]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
