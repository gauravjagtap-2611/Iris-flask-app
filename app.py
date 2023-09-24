import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        
        predicted_class = prediction[0]
        
        return render_template('index.html', prediction_text=f'Class of Flower is {predicted_class}')
    
    except Exception as e:
        return render_template('index.html', error_message=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80,debug=True)