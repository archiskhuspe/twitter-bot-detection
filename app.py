"""Flask web app: serves a form and classifies an account as Bot or Human
using the Decision Tree model trained by model.py (model.pkl)."""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if output == 1:
        output = "Bot"
    else:
        output = "Human"

    return render_template('index.html', prediction_text='Is it a bot, or a human?: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
