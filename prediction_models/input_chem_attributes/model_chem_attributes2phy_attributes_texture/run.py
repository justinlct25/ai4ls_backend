import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load the trained SVR models
svr_models = {}
phy_attributes = ['Coarse', 'Clay', 'Silt', 'BD 0-20']
for attribute in phy_attributes:
    filename = f"svr_model_physical_to_{attribute}.pkl"
    svr_models[attribute] = joblib.load(filename)


new_sample_physical = pd.DataFrame({
    'pH_H2O': [8.15],
    'EC': [30.9],
    'OC': [21.8],
    'CaCO3': [112],
    'P': [22.3],
    'N': [2],
    'K': [220.2]
})

# Standardize the features
scaler = StandardScaler()
new_sample_scaled = scaler.fit_transform(new_sample_physical)

# Make predictions using the loaded models
predictions = {}
for attribute, model in svr_models.items():
    prediction = model.predict(new_sample_scaled)
    predictions[attribute] = prediction[0]

# Display the predictions
for attribute, value in predictions.items():
    print(f"Predicted {attribute}: {value}")
