import pandas as pd
import joblib

# Load the trained model
rf_model_erosion = joblib.load('rf_model_erosion_probability.pkl')

# Separate features (X) for prediction
X_new_data = pd.DataFrame({
    'pH_H2O': [8.15],
    'EC': [22.6],
    'OC': [22.0],
    'CaCO3': [39.0],
    'P': [10.6],
    'N': [2.1],
    'K': [526.7],
    'BD 0-20': [0.964]
})

# X_new_data = pd.DataFrame({
#     'pH_H2O': [6.31],
#     'EC': [11.76],
#     'OC': [15.1],
#     'CaCO3': [0],
#     'P': [19.5],
#     'N': [1.5],
#     'K': [185.9],
#     'BD 0-10': [0.994]
# })

# Predict the probability of erosion for the new data
y_prob_new_data = rf_model_erosion.predict_proba(X_new_data)[:, 1]

# Display the predicted probabilities
print('Predicted Probabilities of Erosion for New Data:')
print(y_prob_new_data)
