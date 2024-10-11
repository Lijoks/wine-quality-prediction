import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data
data = pd.read_csv("data/winequality-red.csv", sep=';')

# Prepare the data
X = data.drop(columns="quality")
y = data['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=80)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'wine_quality_model.pkl')

print("Model trained and saved successfully.")

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('wine_quality_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html file for the homepage

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = request.form.getlist('features')  # Assuming features are passed as a list
    wine_data = np.array([list(map(float, input_data))])  # Convert input to float and reshape
    
    # Make predictions
    prediction = model.predict(wine_data)
    
    return jsonify({'predicted_quality': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)