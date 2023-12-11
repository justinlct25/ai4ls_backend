import pandas as pd
import joblib

# Load the trained SVM model from the saved .pkl file
loaded_model_task1 = joblib.load('svm_model_attributes_managed_l.pkl')

# Assuming you have a new data point stored in a pandas DataFrame called 'new_data_task1'
# 4.85	12.53	47.5	1	12.3	3.1	114.8 (Forestry)
# new_data_task1 = pd.DataFrame({
#     'pH_H2O': [4.85],
#     'EC': [12.53],
#     'OC': [47.5],
#     'CaCO3': [1],
#     'P': [12.3],
#     'N': [3.1],
#     'K': [114.8]
# })
# 8.15	30.9	21.8	112	22.3	2	220.2 (Agriculture)
new_data_task1 = pd.DataFrame({
    'pH_H2O': [8.15],
    'EC': [30.9],
    'OC': [21.8],
    'CaCO3': [112],
    'P': [22.3],
    'N': [2],
    'K': [220.2]
})


# Use the loaded model to make predictions
predicted_land_use_task1 = loaded_model_task1.predict(new_data_task1)
probability_estimates_task1 = loaded_model_task1.predict_proba(new_data_task1)


# Print the predicted land use class
print(f'Managed land? {predicted_land_use_task1[0]}')
print(f'Probability estimates: {probability_estimates_task1[0][predicted_land_use_task1[0]]}')

