import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
parkinsons_data = pd.read_csv('Dataset/parkinson_data.csv')

# Remove non-numeric columns
parkinsons_data = parkinsons_data.drop(columns=['name'], axis=1)

# Checking for null values
print("Null values:\n", parkinsons_data.isnull().sum())

# Display basic info
print(parkinsons_data.info())

# Display class distribution
print(parkinsons_data['status'].value_counts())

# Group by 'status' but only for numeric columns
print(parkinsons_data.groupby('status').mean(numeric_only=True))

# Splitting features and target
X = parkinsons_data.drop(columns=['status'], axis=1)
Y = parkinsons_data['status']

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data:', test_data_accuracy)

# Predicting for a new input sample
input_data = (119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
              0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
              0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654)

# Convert input to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Standardize the input
std_data = scaler.transform(pd.DataFrame(input_data_as_numpy_array, columns=X.columns))


# Make prediction
prediction = model.predict(std_data)
print("Prediction:", prediction)

# Display result
if prediction[0] == 0:
    print("The Person does not have Parkinson’s Disease")
else:
    print("The Person has Parkinson’s Disease")

# Save the model
filename = 'Model/parkinsons_model.sav'
pickle.dump((model, scaler), open(filename, 'wb'))

# Load and test saved model
loaded_model, loaded_scaler = pickle.load(open('Model/parkinsons_model.sav', 'rb'))

# Print feature names
print("Feature names:")
for column in X.columns:
    print(column)

# Print sklearn version
import sklearn
print("Scikit-learn version:", sklearn.__version__)
