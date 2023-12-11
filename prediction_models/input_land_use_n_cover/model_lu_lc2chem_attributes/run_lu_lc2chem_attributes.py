import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
import joblib

# Load the trained SVR models
svr_models = {}
attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']
for attribute in attributes:
    filename = f"svr_model_lu&c_{attribute}.pkl"
    svr_models[attribute] = joblib.load(filename)

# Load label encoders
label_encoder_lu1 = LabelEncoder()
label_encoder_lu1.classes_ = joblib.load("label_encoder_lu1_classes.pkl")

label_encoder_lc0 = LabelEncoder()
label_encoder_lc0.classes_ = joblib.load("label_encoder_lc0_classes.pkl")

# Create a new sample for prediction
new_sample = pd.DataFrame({
    "LU1_Desc_encoded": [label_encoder_lu1.transform(["Forestry"])[0]],
    "LC0_Desc_encoded": [label_encoder_lc0.transform(["Water"])[0]],
})

# Standardize the features
scaler = StandardScaler()
new_sample_scaled = scaler.fit_transform(new_sample)

# Make predictions using the loaded models
predictions = {}
for attribute, model in svr_models.items():
    prediction = model.predict(new_sample_scaled)
    predictions[attribute] = prediction[0]

# Display the predictions
for attribute, value in predictions.items():
    print(f"Predicted {attribute}: {value}")
