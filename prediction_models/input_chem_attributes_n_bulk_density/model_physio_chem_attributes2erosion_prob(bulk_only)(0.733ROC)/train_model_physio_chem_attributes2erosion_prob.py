import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Load the dataset
data = pd.read_csv('../../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion).csv')

soil_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K', 'BD 0-20']

# Define the target variable as the presence or absence of erosion
data['Erosion'] = data[['EROSION_SHEET', 'EROSION_RILL', 'EROSION_GULLY', 'EROSION_MASS', 'EROSION_DEP', 'EROSION_WIND']].max(axis=1)

data = data.dropna(subset=['BD 0-20'], how='any')

print(data.shape[0])

# Replace NaN values with 0 in all columns
data.fillna(0, inplace=True)

# Replace values below LOD in specified columns
for column in soil_attributes:
    data[column] = data[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Filter rows where either the erosion column or the 'BD' column has non-missing values
# data = data.dropna(subset=['Erosion', 'BD 0-10'], how='any')

# Separate features (X) and target variable (y) for erosion prediction
X_erosion = data[soil_attributes]
y_erosion = data['Erosion']

# Split the data into training and testing sets
X_train_erosion, X_test_erosion, y_train_erosion, y_test_erosion = train_test_split(X_erosion, y_erosion, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_model_erosion = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_erosion.fit(X_train_erosion, y_train_erosion)

# Save the trained model to a file
joblib.dump(rf_model_erosion, 'rf_model_erosion_probability.pkl')

# Predict the probability of erosion for the test set
y_prob_erosion = rf_model_erosion.predict_proba(X_test_erosion)[:, 1]

# Evaluate the model using ROC-AUC score
roc_auc = roc_auc_score(y_test_erosion, y_prob_erosion)

# Print the results
print(f'ROC-AUC for Probability of Erosion: {roc_auc}')

accuracy_filename = 'erosion_prob_model_accuracy.txt'
with open(accuracy_filename, 'w') as file:
    file.write(f'ROC-AUC for Probability of Erosion: {roc_auc}')
