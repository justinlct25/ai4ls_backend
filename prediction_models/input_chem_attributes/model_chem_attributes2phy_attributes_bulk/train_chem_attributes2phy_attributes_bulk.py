# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib  

# Load your dataset from the CSV file
df = pd.read_csv("../../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion)(out-standard)(textural-info).csv")

# Replace values below LOD in specified columns
chem_attribute_columns = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']
for column in chem_attribute_columns:
    df[column] = df[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)
phy_attribute_columns = ['BD 0-10', 'BD 10-20', 'BD 0-20']


# Filter out rows with any empty values
df_filtered = df.dropna(subset=phy_attribute_columns, inplace=True)
df.fillna(0, inplace=True)
print(f"Number of rows in the original dataframe: {df.shape[0]}")


# Extract features (X) and target variables (y)
chemical_features = df[chem_attribute_columns]
physical_targets = df[phy_attribute_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(chemical_features, physical_targets, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train separate SVR models for each target variable (chemical attribute)
svr_models = {}
accuracy_file = "svr_models_accuracies.txt"
with open(accuracy_file, "w") as file:
    for attribute in phy_attribute_columns:
        svr = SVR()
        svr.fit(X_train_scaled, y_train[attribute])
        svr_models[attribute] = svr
        predictions = svr.predict(X_test_scaled)  # Make predictions on the test set
        mae = mean_absolute_error(y_test[attribute], predictions)  # Evaluate the model
        accuracy_info = f"MAE for {attribute}: {mae}"
        print(accuracy_info)
        file.write(accuracy_info + "\n")

# Save each model to a .pkl file
for attribute, model in svr_models.items():
    filename = f"svr_model_physical_to_{attribute}.pkl"
    if filename:
        joblib.dump(model, filename)

# Now, you can use the trained models for making predictions on new data.
# For example, to predict chemical attributes for a new sample:
new_sample_physical = pd.DataFrame({
    'pH_H2O': [8.15],
    'EC': [30.9],
    'OC': [21.8],
    'CaCO3': [112],
    'P': [22.3],
    'N': [2],
    'K': [220.2]
})

new_sample_physical_scaled = scaler.transform(new_sample_physical.values.reshape(1, -1))
chemical_predictions = {attribute: model.predict(new_sample_physical_scaled)[0] for attribute, model in svr_models.items()}
print("Predictions for new sample (chemical attributes):", chemical_predictions)


new_sample_physical = pd.DataFrame({
    'pH_H2O': [4.15],
    'EC': [10.9],
    'OC': [11.8],
    'CaCO3': [152],
    'P': [19.3],
    'N': [2.5],
    'K': [120.2]
})

new_sample_physical_scaled = scaler.transform(new_sample_physical.values.reshape(1, -1))
chemical_predictions = {attribute: model.predict(new_sample_physical_scaled)[0] for attribute, model in svr_models.items()}
print("Predictions for new sample (chemical attributes):", chemical_predictions)

