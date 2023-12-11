import pandas as pd
import joblib

# Load the trained model
rf_model_erosion = joblib.load('rf_model_erosion_probability.pkl')

# # Separate features (X) for prediction
# X_new_data = pd.DataFrame({
#     'pH_H2O': [8.15],
#     'EC': [30.9],
#     'OC': [21.8],
#     'CaCO3': [112],
#     'P': [22.3],
#     'N': [2],
#     'K': [220.2],
# })

# Separate features (X) for prediction
X_new_data = pd.DataFrame({
    'pH_H2O': [4.49],
    'EC': [12.48],
    'OC': [108.7],
    'CaCO3': [1],
    'P': [10.7],
    'N': [4.7],
    'K': [155.2],
})


# Predict the probability of erosion for the new data
y_prob_new_data = rf_model_erosion.predict_proba(X_new_data)[:, 1]

# Display the predicted probabilities
print('Predicted Probabilities of Erosion for New Data:')
print(y_prob_new_data)
