import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load dataset
print("All libraries imported successfully!")
heart_data = pd.read_csv('C:\Users\HASSAN KHAN\Desktop')

# Inspect data
print(heart_data.head())
print(heart_data.tail())
print(heart_data.info())
print(heart_data.describe())

# Correlation Matrix
correlation_matrix = heart_data.corr()
print(correlation_matrix)
# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Output values Count
heart_data['target'].value_counts()

# Split data into features and labels
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data:', testing_data_accuracy)

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as 'heart_disease_model.pkl'")

# Model Checking
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print('The person does not have heart disease')
else:
    print('The person has heart disease')