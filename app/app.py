from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return '''
        <h2>Machine Learning Model Deployment</h2>
        <form action="/predict" method="post">
            <input type="text" name="feature1" placeholder="Feature 1" required><br>
            <input type="text" name="feature2" placeholder="Feature 2" required><br>
            <input type="text" name="feature3" placeholder="Feature 3" required><br>
            <button type="submit">Predict</button>
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data
        features = [float(x) for x in request.form.values()]
        data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(data)[0]

        return f"<h3>Prediction: {prediction}</h3><a href='/'>Go Back</a>"

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>Go Back</a>"

# Note: Do not include app.run() here (Render runs via run.py)
