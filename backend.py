from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
CORS(app, origins="*")


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
    
def load_scaler(scaler_path):
    try:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model file not found at: {scaler_path}")
        model = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}.")
        return model
    except Exception as e:
        print(f"Error loading the scaler from {scaler_path}: {e}")
        return None

soil_units = {}
with open('./soil_attributes_units.json') as json_file:
    soil_units = json.load(json_file)

soil_standards = {}
with open('./soil_attributes_standards.json') as json_file:
    soil_standards = json.load(json_file)

soil_attributes_info = {}
with open('./soil_attributes_info.json') as json_file:
    soil_attributes_info = json.load(json_file)

def is_soil_attribute_hv_standard(attribute):
    return True if attribute in soil_standards else False

def is_soil_attribute_out_standard(attribute, value):
    attr_range = soil_standards[attribute]
    if "min" in attr_range and value < attr_range["min"]:
        return False
    elif "max" in attr_range and value > attr_range["max"]:
        return False
    else:
        return True
    
loaded_models = {}
with open('./models_config.json', 'r') as json_file:
    loaded_models = json.load(json_file)
    for input, prediction in loaded_models["inputs"].items():
        for output, output_info in prediction["outputs"].items():
            model_folder_path = output_info["folder"]
            for model, model_info in output_info["models"].items():
                model_file_path = os.path.join(model_folder_path, model_info["model_file"])
                model_info["model"] = load_model(model_file_path)
                if "scaler_file" in model_info:
                    scaler_file_path = os.path.join(model_folder_path, model_info["scaler_file"])
                    model_info["scaler"] = load_scaler(scaler_file_path)


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
            "model_info": "Support Vector Classification (SVC) model",
            "model_accuracy": prediction_models["is_managed"]["accuracy"]
        }
        prediction_results["erosion_probability"] = {
            "result": {
                "probability": float(prediction_models["erosion_probability"]["model"].predict_proba(soil_attributes)[:, 1])
            },
            "model_info": "Support Vector Classification (SVC) model",
            "model_accuracy": prediction_models["erosion_probability"]["accuracy"]
        }
        # physical attribute texture model processing
        prediction_results["phy_attributes_texture"] = {
            "result": {}, 
            "model_info": "Support Vector Regression (SVR) models",
            "model_accuracy": {}
        }
        for attribute, model_info in prediction_models["phy_attributes_texture"].items():
            value = model_info["model"].predict(soil_attributes_scaled)[0]
            attribute_result = {}
            attribute_result["value"] = value
            if is_soil_attribute_hv_standard(attribute):
                attribute_result["out_of_standard"] = is_soil_attribute_out_standard(attribute, value)
            attribute_result["info"] = soil_attributes_info[attribute]
            prediction_results["phy_attributes_texture"]["result"][attribute] = attribute_result
            prediction_results["phy_attributes_texture"]["model_accuracy"][attribute] = model_info["accuracy"]
        # physical attribute bulk density model processing
        prediction_results["phy_attributes_bulk_density"] = {
            "result": {}, 
            "model_info": "Support Vector Regression (SVR) models",
            "model_accuracy": {}
        }
        for attribute, model_info in prediction_models["phy_attributes_bulk_density"].items():
            value = model_info["model"].predict(soil_attributes_scaled)[0]
            attribute_result = {}
            attribute_result["value"] = value
            if is_soil_attribute_hv_standard(attribute):
                attribute_result["out_of_standard"] = is_soil_attribute_out_standard(attribute, value)
            attribute_result["info"] = soil_attributes_info[attribute]
            prediction_results["phy_attributes_bulk_density"]["result"][attribute] = attribute_result
            prediction_results["phy_attributes_bulk_density"]["model_accuracy"][attribute] = model_info["accuracy"]
        return jsonify(prediction_results)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/land_use_and_cover_for_predictions', methods=['POST'])
def land_use_and_cover_for_predictions():
    try:
        classes_info = loaded_models["inputs"]["land_use_n_land_cover"]["classes"]
        land_use_classes_file = os.path.join(classes_info["folder"], classes_info["land_use"])
        land_cover_classes_file = os.path.join(classes_info["folder"], classes_info["land_cover"])
        label_encoder_lu1 = LabelEncoder()
        label_encoder_lc0 = LabelEncoder()
        label_encoder_lu1.classes_ = joblib.load(land_use_classes_file)
        label_encoder_lc0.classes_ = joblib.load(land_cover_classes_file)
        data = request.get_json()
        input_classes = pd.DataFrame(data, index=[0])
        input_classes["LU1_Desc_encoded"] = label_encoder_lu1.transform(input_classes["LU1_Desc"])
        input_classes["LC0_Desc_encoded"] = label_encoder_lc0.transform(input_classes["LC0_Desc"])
        input_classes_numeric = input_classes.drop(["LU1_Desc", "LC0_Desc"], axis=1)
        prediction_models = loaded_models["inputs"]["land_use_n_land_cover"]["outputs"]["all_attributes"]["models"]
        prediction_results = {}
        prediction_results["all_attributes"] = {
            "result": {},
            "model_info": "Support Vector Regression (SVR) models",
            "model_accuracy": {}
        }
        for attribute, model_info in prediction_models.items():
            input_classes_scaled = model_info["scaler"].transform(input_classes_numeric)
            value = model_info["model"].predict(input_classes_scaled)[0]
            attribute_result = {}
            attribute_result["value"] = value
            if is_soil_attribute_hv_standard(attribute):
                attribute_result["out_of_standard"] = is_soil_attribute_out_standard(attribute, value)
            attribute_result["info"] = soil_attributes_info[attribute]
            prediction_results["all_attributes"]["result"][attribute] = attribute_result
            prediction_results["all_attributes"]["model_accuracy"][attribute] = model_info["accuracy"]
        
        return jsonify(prediction_results)
    except Exception as e:
        return jsonify({'error': str[e]})
    
@app.route('/land_use_and_cover_classes')
def land_use_and_cover_classes():
    try:
        classes_info = loaded_models["inputs"]["land_use_n_land_cover"]["classes"]
        land_use_classes_file = os.path.join(classes_info["folder"], classes_info["land_use"])
        land_cover_classes_file = os.path.join(classes_info["folder"], classes_info["land_cover"])
        classes = {
            "land_use_classes": joblib.load(land_use_classes_file).tolist(),
            "land_cover_classes": joblib.load(land_cover_classes_file).tolist()
        }
        return jsonify(classes)
    except Exception as e:
        return jsonify({'error': str[e]})
    
@app.route('/soil_attribute_units')
def soil_attribute_units():
    try:
        return jsonify(soil_units)
    except Exception as e:
        return jsonify({'error': str[e]})
    
@app.route('/soil_attribute_standards')
def soil_attribute_standards():
    try:
        return jsonify(soil_standards)
    except Exception as e:
        return jsonify({'error': str[e]})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

