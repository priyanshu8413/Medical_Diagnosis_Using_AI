import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Dataset/diabetes_data.csv')

# Display basic info
data.info()
print(data.head())

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Separate features and target variable
X = data.drop(columns=['Outcome'])  # Assuming 'Outcome' is the target column
Y = data['Outcome']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f'Accuracy on Training data: {train_accuracy * 100:.2f}%')
print(f'Accuracy on Test data: {test_accuracy * 100:.2f}%')

# Save the trained model
import pickle
filename = 'Model/diabetes_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Model saved successfully.")
