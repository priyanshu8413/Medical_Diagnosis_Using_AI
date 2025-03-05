import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

data = pd.read_csv('Dataset/survey lung cancer.csv')
label_encoder = preprocessing.LabelEncoder()

data['GENDER'] = label_encoder.fit_transform(data['GENDER'])

data['GENDER'].unique()
data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])

data['LUNG_CANCER'].unique()
data.to_csv('Dataset/prepocessed_lungs_data.csv')
X = data.drop(columns='LUNG_CANCER', axis=1)
Y = data['LUNG_CANCER']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)
input_data = (0,61,1,1,1,1,2,2,1,1,1,1,2,1,1)

input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# reshape the numpy array as we are predicting for only on instance
input_df = pd.DataFrame(input_data_as_numpy_array, columns=X.columns)
prediction = model.predict(input_df)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Lung Cancer')
else:
  print('The Person has Lung Cancer')
import pickle
filename = 'Model/lungs_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open('Model/lungs_disease_model.sav', 'rb'))
for column in X_train.columns:
  print(column)