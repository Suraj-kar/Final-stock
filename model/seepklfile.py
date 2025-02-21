import joblib

# Load the saved models
scaler = joblib.load("model/saved_models/SCB_scaler.pkl")
model = joblib.load("model/saved_models/SCB_model.pkl")

# Print the contents
print("Scaler contents:")
print("Means:", scaler.means)
print("Standard deviations:", scaler.stds)

print("\nModel contents:")
print("Weights:", model.weights)
print("Bias:", model.bias)