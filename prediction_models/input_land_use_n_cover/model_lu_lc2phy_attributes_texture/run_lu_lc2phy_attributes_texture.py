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

# Load label encoders
label_encoder_lu1 = LabelEncoder()
label_encoder_lu1.classes_ = joblib.load("label_encoder_lu1_classes.pkl")

label_encoder_lc0 = LabelEncoder()
label_encoder_lc0.classes_ = joblib.load("label_encoder_lc0_classes.pkl")

# Create new samples for prediction
new_sample1 = pd.DataFrame({
    "LU1_Desc_encoded": [label_encoder_lu1.transform(["Forestry"])[0]],
    "LC0_Desc_encoded": [label_encoder_lc0.transform(["Woodland"])[0]],
})

new_sample2 = pd.DataFrame({
    "LU1_Desc_encoded": [label_encoder_lu1.transform(["Agriculture (excluding fallow land and kitchen gardens)"])[0]],
    "LC0_Desc_encoded": [label_encoder_lc0.transform(["Bareland"])[0]],
})

# Standardize the features using the same scaler
scaler = StandardScaler()
X_train = pd.concat([new_sample1, new_sample2])  # Combine samples for fitting the scaler
X_train_scaled = scaler.fit_transform(X_train)

# Make predictions using the loaded models
predictions1 = {}
predictions2 = {}

for attribute, model in svr_models.items():
    # Extract the corresponding rows for each sample
    new_sample1_scaled = X_train_scaled[:1, :]
    new_sample2_scaled = X_train_scaled[1:, :]

    prediction1 = model.predict(new_sample1_scaled)
    predictions1[attribute] = prediction1[0]

    prediction2 = model.predict(new_sample2_scaled)
    predictions2[attribute] = prediction2[0]

# Display the predictions
print("prediction1:")
for attribute, value in predictions1.items():
    print(f"Predicted {attribute}: {value}")

print("prediction2:")
for attribute, value in predictions2.items():
    print(f"Predicted {attribute}: {value}")
