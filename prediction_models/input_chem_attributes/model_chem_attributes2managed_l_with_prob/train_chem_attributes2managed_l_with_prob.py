
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib  


data = pd.read_csv('../../csv_processing/LUCAS-SOIL-2018(managed-l).csv')

soil_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']

# Replace NaN values with 0 in all columns
data.fillna(0, inplace=True)

# Replace values below LOD in specified columns
for column in soil_attributes:
    data[column] = data[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Separate features (X) and target variable (y) for Task 1
X_ = data[soil_attributes]
y_ = data['Un-/Managed_LU']

# Split the data into training and testing sets
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', verbose=True, probability=True)
svm_model.fit(X_train_, y_train_)

# Save the trained model to a file
joblib.dump(svm_model, 'svm_model_attributes_managed_l.pkl')

# Make predictions on the test set
y_pred_ = svm_model.predict(X_test_)

# Evaluate the model
accuracy = accuracy_score(y_test_, y_pred_)
classification_report_ = classification_report(y_test_, y_pred_)

report_text = f'Accuracy for managed land detection: {accuracy}\nClassification Report for Task 1:\n{classification_report_}'
with open('training_report_.txt', 'w') as file:
    file.write(report_text)

print(report_text)

