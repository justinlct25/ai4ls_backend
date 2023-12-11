from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")
        return None

loaded_models = {}
with open('./models_config.json', 'r') as json_file:
    loaded_models = json.load(json_file)
    for input, prediction in loaded_models["inputs"].items():
        for output, output_info in prediction["outputs"].items():
            model_folder_path = output_info["folder"]
            for model, model_info in output_info["models"].items():
                model_file_path = os.path.join(model_folder_path, model_info["model_file"])
                model_info["model"] = load_model(model_file_path)

@app.route('/predict_land_management', methods=['POST'])
def predict_land_management():
    try:
        data = request.get_json()
        soil_attributes = pd.DataFrame(data, index=[0])
        model = loaded_models["inputs"]["chem_attributes"]["outputs"]["is_managed_land_n_probability"]["models"]["Is_managed"]["model"]
        prediction = model.predict(soil_attributes)
        is_managed = int(prediction[0])
        probability = model.predict_proba(soil_attributes)[0][is_managed]
        return jsonify({"prediction": is_managed, "probability": probability})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/chem_attributes_for_predictions', methods=['POST'])
def chem_attributes_for_predictions():
    try:
        data = request.get_json()
        soil_attributes = pd.DataFrame(data, index=[0])
        scaler = StandardScaler()
        soil_attributes_scaled = scaler.fit_transform(soil_attributes)
        prediction_results = {}
        prediction_models = {}
        prediction_models["is_managed"] = loaded_models["inputs"]["chem_attributes"]["outputs"]["is_managed_land_n_probability"]["models"]["Is_managed"]
        prediction_models["erosion_probability"] = loaded_models["inputs"]["chem_attributes"]["outputs"]["erosion_probability"]["models"]["Erosion_prob"]
        prediction_models["phy_attributes_texture"] = loaded_models["inputs"]["chem_attributes"]["outputs"]["physical_attributes_texture"]["models"]
        prediction_models["phy_attributes_bulk_density"] = loaded_models["inputs"]["chem_attributes"]["outputs"]["physical_attributes_bulk_density"]["models"]
        is_managed = int(prediction_models["is_managed"]["model"].predict(soil_attributes))
        prediction_results["is_managed"] = {
            "result": {
                "prediction": is_managed,
                "probability": prediction_models["is_managed"]["model"].predict_proba(soil_attributes)[0][is_managed],
            },
            "model_accuracy": prediction_models["is_managed"]["accuracy"]
        }
        prediction_results["erosion_probability"] = {
            "result": {
                "probability": float(prediction_models["erosion_probability"]["model"].predict_proba(soil_attributes)[:, 1])
            },
            "model_accuracy": prediction_models["erosion_probability"]["accuracy"]
        }
        prediction_results["phy_attributes_texture"] = {
            "result": {}, 
            "model_accuracy": {}
        }
        for attribute, model_info in prediction_models["phy_attributes_texture"].items():
            prediction_results["phy_attributes_texture"]["result"][attribute] = model_info["model"].predict(soil_attributes_scaled)[0]
            prediction_results["phy_attributes_texture"]["model_accuracy"][attribute] = model_info["accuracy"]
        prediction_results["phy_attributes_bulk_density"] = {
            "result": {}, 
            "model_accuracy": {}
        }
        for attribute, model_info in prediction_models["phy_attributes_bulk_density"].items():
            prediction_results["phy_attributes_bulk_density"]["result"][attribute] = model_info["model"].predict(soil_attributes_scaled)[0]
            prediction_results["phy_attributes_bulk_density"]["model_accuracy"][attribute] = model_info["accuracy"]
        return jsonify(prediction_results)
    except Exception as e:
        return jsonify({'error': str(e)})
    
label_encoder_lu1 = LabelEncoder()
label_encoder_lc0 = LabelEncoder()

@app.route('/land_use_and_cover_for_predictions', methods=['POST'])
def land_use_and_cover_for_predictions():
    try:
        classes_info = loaded_models["inputs"]["land_use_n_land_cover"]["classes"]
        land_use_classes_file = os.path.join(classes_info["folder"], classes_info["land_use"])
        land_cover_classes_file = os.path.join(classes_info["folder"], classes_info["land_cover"])
        label_encoder_lu1.classes_ = joblib.load(land_use_classes_file)
        label_encoder_lc0.classes_ = joblib.load(land_cover_classes_file)
        data = request.get_json()
        input_classes = pd.DataFrame(data, index=[0])
        input_classes["LU1_Desc_encoded"] = label_encoder_lu1.transform(input_classes["LU1_Desc"])
        input_classes["LC0_Desc_encoded"] = label_encoder_lc0.transform(input_classes["LC0_Desc"])
        input_classes_numeric = input_classes.drop(["LU1_Desc", "LC0_Desc"], axis=1)
        scaler = StandardScaler()
        input_classes_scaled = scaler.fit_transform(input_classes_numeric)
        prediction_models = loaded_models["inputs"]["land_use_n_land_cover"]["outputs"]["all_attributes"]["models"]
        prediction_results = {}
        for attribute, model_info in prediction_models.items():
            prediction_results[attribute] = model_info["model"].predict(input_classes_scaled)[0]
        return jsonify(prediction_results)
    except Exception as e:
        return jsonify({'error': str[e]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

@app.route('/lucas_sample_points')
def get_lucas_sample_points():
    return "Hello"