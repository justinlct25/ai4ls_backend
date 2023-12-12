import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
import joblib

# Load the trained SVR models
svr_models = {}
attributes = ['Coarse', 'Clay', 'Sand', 'Silt']
for attribute in attributes:
    filename = f"svr_model_lu&c_{attribute}.pkl"
    svr_models[attribute] = joblib.load(filename)

label_encoder_lu1 = LabelEncoder()
label_encoder_lu1.classes_ = joblib.load("label_encoder_lu1_classes.pkl")
label_encoder_lc0 = LabelEncoder()
label_encoder_lc0.classes_ = joblib.load("label_encoder_lc0_classes.pkl")

# new_data = pd.DataFrame({
#     "LU1_Desc_encoded": [label_encoder_lu1.transform(["Forestry"])[0]],
#     "LC0_Desc_encoded": [label_encoder_lc0.transform(["Woodland"])[0]],
# })

new_data = pd.DataFrame({
    "LU1_Desc_encoded": [label_encoder_lu1.transform(["Agriculture (excluding fallow land and kitchen gardens)"])[0]],
    "LC0_Desc_encoded": [label_encoder_lc0.transform(["Bareland"])[0]],
})

# Load the saved StandardScaler
scaler = joblib.load("standard_scaler.pkl")
# new_data_scaled = scaler.fit_transform(new_data)

# Standardize the features using the loaded scaler
new_data_scaled = scaler.transform(new_data)

predictions = {}
for attribute, model in svr_models.items():
    predictions[attribute] = model.predict(new_data_scaled.reshape(1, -1))[0]

for attribute, value in predictions.items():
    print(f"Predicted {attribute}: {value}")

# `predictions` now contains the predicted values for each attribute
