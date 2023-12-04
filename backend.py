from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib

app = Flask(__name__)


@app.before_first_request
def load_model():
    global model
    try:
        model_path = './svm_model_attributes_managed_l.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = joblib.load(model_path)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading the model: {e}")

@app.route('/predict_land_management', methods=['POST'])
def predict_land_management():
    try:
        data = request.get_json()
        soil_attributes = pd.DataFrame(data, index=[0])
        prediction = model.predict(soil_attributes)
        is_managed = int(prediction[0])
        return jsonify({'prediction': is_managed})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

@app.route('/lucas_sample_points')
def get_lucas_sample_points():
    return "Hello"