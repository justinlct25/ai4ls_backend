# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib  


# Load your dataset from the CSV file
df = pd.read_csv("../../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion)(out-standard)(textural-info).csv")

attribute_columns = ['BD 0-10', 'BD 10-20', 'BD 0-20']

df.dropna(subset=attribute_columns, inplace=True)
print(f"Number of rows in the original dataframe: {df.shape[0]}")

# Extract features (X) and target variables (y)
features = df[["LU1_Desc", "LC0_Desc"]]
targets = df[attribute_columns]

# Encode categorical variables (assuming LU1_desc and LC0_Desc are categorical)
label_encoder_lu1 = LabelEncoder()
features["LU1_Desc_encoded"] = label_encoder_lu1.fit_transform(features["LU1_Desc"])
label_encoder_lc0 = LabelEncoder()
features["LC0_Desc_encoded"] = label_encoder_lc0.fit_transform(features["LC0_Desc"])
joblib.dump(label_encoder_lu1.classes_, "label_encoder_lu1_classes.pkl")
joblib.dump(label_encoder_lc0.classes_, "label_encoder_lc0_classes.pkl")

# Drop the original categorical columns
features = features.drop(["LU1_Desc", "LC0_Desc"], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train separate SVR models for each target variable
svr_models = {}
accuracy_file = "svr_models_accuracies.txt"
with open(accuracy_file, "w") as file:
    for attribute in targets.columns:
        svr = SVR()
        svr.fit(X_train_scaled, y_train[attribute])
        svr_models[attribute] = svr
        predictions = svr.predict(X_test_scaled) # Make predictions on the test set
        mae = mean_absolute_error(y_test[attribute], predictions) # Evaluate the model
        accuracy_info = f"MAE for {attribute}: {mae}"
        print(accuracy_info)
        file.write(accuracy_info + "\n")

# Save each model to a .pkl file
for attribute, model in svr_models.items():
    filename = f"svr_model_lu&c_{attribute}.pkl"
    if filename:
        joblib.dump(model, filename)

# # Now, you can use the trained models for making predictions on new data.
# # For example, to predict the chemical attributes for a new sample:
# new_sample = pd.DataFrame({
#     "LU1_Desc_encoded": [label_encoder_lu1.transform(["Forestry"])[0]],
#     # "LC0_Desc_encoded": [label_encoder_lc0.transform(["Woodland"])[0]],
#     "LC0_Desc_encoded": [label_encoder_lc0.transform(["Water"])[0]],
# })

# new_sample_scaled = scaler.transform(new_sample)
# predictions = {target: model.predict(new_sample_scaled.reshape(1, -1))[0] for target, model in svr_models.items()}
# print("Predictions for new sample:", predictions)
