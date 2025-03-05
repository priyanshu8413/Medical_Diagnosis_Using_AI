import sklearn
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load Dataset
heart_data = pd.read_csv('Dataset/heart_disease_data.csv')

# Split Data into Features & Labels
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# **Apply Feature Scaling**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, Y_train)

# Predictions
X_train_prediction = model.predict(X_train_scaled)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data :', training_data_accuracy)

X_test_prediction = model.predict(X_test_scaled)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data :', test_data_accuracy)

# **Test with New Input Data**
input_data = (57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# **Apply Scaling to Input Data**
input_data_df = pd.DataFrame(input_data_as_numpy_array, columns=X.columns)
input_data_scaled = scaler.transform(input_data_df)


# Make Prediction
prediction = model.predict(input_data_scaled)
print('The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have Heart Disease')

# Save the Model
filename = 'Model/heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the Model
loaded_model = pickle.load(open(filename, 'rb'))

# Print Feature Names
for column in X.columns:
    print(column)

# Print sklearn Version
print(sklearn.__version__)
