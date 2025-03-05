import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load preprocessed dataset
file_path = "Dataset/prepocessed_hypothyroid.csv"
df = pd.read_csv(file_path)

# Define input features and target variable
x = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(x_train, y_train)

# Evaluate model performance
train_accuracy = accuracy_score(model.predict(x_train), y_train)
test_accuracy = accuracy_score(model.predict(x_test), y_test)

print(f'Accuracy on Training Data: {train_accuracy:.4f}')
print(f'Accuracy on Test Data: {test_accuracy:.4f}')

# Save the trained model
pickle.dump(model, open("Model/Thyroid_model.sav", "wb"))

# Load and use the model for prediction
loaded_model = pickle.load(open("Model/Thyroid_model.sav", "rb"))

# Example input data for prediction (must match the feature count)
input_data = [44, 0, 0, 1.4, 1, 2.5, 130]
input_data_df = pd.DataFrame([input_data], columns=x_train.columns)

# Make a prediction
prediction = loaded_model.predict(input_data_df)

# Output result
if prediction[0] == 0:
    print('The Person does not have a HyperThyroid Disease')
else:
    print('The Person has HyperThyroid Disease')
