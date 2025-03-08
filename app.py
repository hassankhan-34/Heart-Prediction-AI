from flask import Flask, request, render_template_string, session, redirect, url_for
import numpy as np
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Create Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# HTML Template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        form {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            width: 300px;
        }
        h1 {
            text-align: center;
            color: #343a40;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #495057;
        }
        input, button {
            width: 100%;
            padding: 8px;
            margin-bottom: 16px;
            border: 1px solid #ced4da;
            border-radius: 4px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <form action="{{ next_url }}" method="post">
        <h1>Heart Disease Prediction</h1>
        <label for="value">{{ question }}</label>
        <input type="number" id="value" name="value" required {% if step == 9 %}step="0.1"{% endif %}>
        <button type="submit">Next</button>
        
        {% if prediction_text %}
        <p><strong>Result:</strong> {{ prediction_text }}</p>
        {% endif %}
    </form>
</body>
</html>
"""

# Question sequence
questions = [
    "Enter Age:",
    "Enter Sex (1 = Male, 0 = Female):",
    "Enter Chest Pain Type (0-3):",
    "Enter Resting Blood Pressure:",
    "Enter Cholesterol:",
    "Enter Fasting Blood Sugar (1 = High, 0 = Normal):",
    "Enter Resting ECG Results (0-2):",
    "Enter Maximum Heart Rate Achieved:",
    "Enter Exercise-Induced Angina (1 = Yes, 0 = No):",
    "Enter ST Depression Induced by Exercise:",
    "Enter Slope of Peak Exercise ST Segment (0-2):",
    "Enter Number of Major Vessels (0-4):",
    "Enter Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect):"
]

@app.route('/')
def home():
    # Initialize session for multi-step form
    session['inputs'] = []
    session['step'] = 0
    return redirect(url_for('step'))

@app.route('/step', methods=['GET', 'POST'])
def step():
    step = session.get('step', 0)
    inputs = session.get('inputs', [])
    
    if request.method == 'POST' and step < len(questions):
        # Collect the current step's input
        inputs.append(float(request.form['value']))
        session['inputs'] = inputs
        session['step'] = step + 1
    
    if step < len(questions):
        return render_template_string(
            html_template,
            question=questions[step],
            step=step,
            next_url=url_for('step')
        )
    else:
        # Perform prediction once all inputs are collected
        input_data = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_data)
        prediction_text = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"
        
        # Reset session after prediction
        session.clear()
        return render_template_string(
            html_template,
            prediction_text=prediction_text,
            question="Prediction complete. See the result below:",
            step=step,
            next_url=url_for('home')
        )

if __name__ == '__main__':
    app.run(debug=True)
