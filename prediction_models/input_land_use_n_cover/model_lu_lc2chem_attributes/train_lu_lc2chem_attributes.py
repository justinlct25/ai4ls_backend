# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib  


# Load your dataset from the CSV file
df = pd.read_csv("../../csv_processing/LUCAS-SOIL-2018.csv")

# Replace NaN values with 0 in all columns
df.fillna(0, inplace=True)

# Replace values below LOD in specified columns
attribute_columns = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']
for column in attribute_columns:
    df[column] = df[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Extract features (X) and target variables (y)
features = df[["LU1_Desc", "LC0_Desc"]]
targets = df[["pH_H2O", "EC", "OC", "CaCO3", "P", "N", "K"]]

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

# Save the fitted scaler
scaler_filename = "standard_scaler_lu_lc2chem_attributes.pkl"
joblib.dump(scaler, scaler_filename)

# Save each model to a .pkl file
for attribute, model in svr_models.items():
    filename = f"svr_model_lu&c_{attribute}.pkl"
    if filename:
        joblib.dump(model, filename)
