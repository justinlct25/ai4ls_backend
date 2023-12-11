import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
import joblib

# Load the trained SVR models
svr_models = {}
chem_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']
phy_texture_attributes = ['Coarse', 'Clay', 'Sand', 'Silt', 'BD 0-20']
phy_bulk_density = ['BD 0-10', 'BD 10-20', 'BD 0-20']
all_attributes = chem_attributes + phy_texture_attributes + phy_bulk_density
for attribute in all_attributes:
    filename = f"svr_model_lu&c_{attribute}.pkl"
    svr_models[attribute] = joblib.load(filename)

# Load label encoders
label_encoder_lu1 = LabelEncoder()
label_encoder_lu1.classes_ = joblib.load("label_encoder_lu1_classes.pkl")

label_encoder_lc0 = LabelEncoder()
label_encoder_lc0.classes_ = joblib.load("label_encoder_lc0_classes.pkl")

# Create a new sample for prediction
new_sample1 = pd.DataFrame({
    "LU1_Desc": ["Forestry"],
    "LC0_Desc": ["Woodland"],
})

new_sample2 = pd.DataFrame({
    "LU1_Desc": ["Agriculture (excluding fallow land and kitchen gardens)"],
    "LC0_Desc": ["Bareland"],
})

# Encode categorical variables
new_sample1["LU1_Desc_encoded"] = label_encoder_lu1.transform(new_sample1["LU1_Desc"])
new_sample1["LC0_Desc_encoded"] = label_encoder_lc0.transform(new_sample1["LC0_Desc"])

# Encode categorical variables
new_sample2["LU1_Desc_encoded"] = label_encoder_lu1.transform(new_sample2["LU1_Desc"])
new_sample2["LC0_Desc_encoded"] = label_encoder_lc0.transform(new_sample2["LC0_Desc"])


# # Standardize the features
# scaler = StandardScaler()
# new_sample_scaled = scaler.fit_transform(new_sample[['LU1_Desc_encoded', 'LC0_Desc_encoded']])

# Standardize the features using the same scaler
scaler = StandardScaler()
X_train = pd.concat([new_sample1, new_sample2])
X_train_numeric = X_train.drop(["LU1_Desc", "LC0_Desc"], axis=1)

# Standardize the numeric features
X_train_scaled = scaler.fit_transform(X_train_numeric)

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
