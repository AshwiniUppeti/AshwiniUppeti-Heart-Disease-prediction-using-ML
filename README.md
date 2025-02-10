# AshwiniUppeti-Heart-Disease-prediction-using-ML
# Heart Disease Prediction using Logistic Regression

## Overview
This project implements a heart disease prediction model using Logistic Regression. The model is trained on a dataset containing medical information related to heart disease and predicts whether a person has heart disease based on given input parameters.

## Dataset
The dataset used for this project is assumed to be in CSV format and contains various medical parameters such as age, cholesterol levels, blood pressure, and more. The target column represents whether the patient has heart disease (1) or not (0).

## Requirements
To run this project, you need to install the following dependencies:

```bash
pip install numpy pandas scikit-learn
```

## Code Explanation
The script follows these steps:

1. Load the dataset using Pandas.
2. Explore and preprocess the data.
3. Split the dataset into training and testing sets.
4. Train a Logistic Regression model on the training data.
5. Evaluate the model using accuracy score.
6. Predict heart disease for a given set of input parameters.

## Usage

Run the Python script to train and test the model:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
heart_data = pd.read_csv('/content/heart (1).csv')

# Data preprocessing
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data:', test_data_accuracy)

# Making a prediction
input_data = (62, 0, 0, 138, 294, 1, 1, 106, 0, 1.9, 1, 3, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')
```

## Expected Output
After training the model, you will see accuracy scores for both training and testing datasets. When you input patient data, the model will predict whether the person has heart disease or not.

## Notes
- Ensure that the dataset is available at the specified location before running the script.
- The model may require further tuning for improved accuracy.

## License
This project is open-source and free to use for educational purposes.

