import sys
import pickle

# Load the label encoder for categorical features
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Check the Label Encoder's Classes
print("Label Encoder Classes:", label_encoder.classes_)

# Compare against original training data
# For example, if 'Gender' was one of the encoded features
# print("Unique values in original 'Gender' column:", original_training_data['Gender'].unique())
