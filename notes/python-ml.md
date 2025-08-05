# MACHINE LEARNING NOTES

## DATA PREPROCESSING 
-import the data 
- clean the date 
- split into training and test sets 

## MODELLING 
- build the model
- train the model 
- make predictions 

## EVALUATIONS
- calculate preformance metrics
- make verdict 

## FEATURE SCALING 
- applied to the columns in datasets 
- normalization 
- standardization 

### DATA PREPREOCESSING STEPS
#Importing the necessary libraries 

import pandas as pd
import numpy as np
import sklearn.model_selection as train_test_split

#Loding the Iris dataset 

dataset = pd.read_csv('iris.csv')

#Creating the matrix of features (X) and the dependent variable vector (y)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Printing the matrix of features and the dependent variable vector 

print(X)

print(y)

---

Taking Care of Missing Data 

#Importing the necessary libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

#Load the dataset 

df = pd.read_csv('pima-indians-diabetes.csv')

#Identify missing data (assumes that missing data is represented as NaN)

missing_data = df.isnull().sum()

#Print the number of missing entries in each column 

print(”Missing data: \n”, missing_data)

#Configure an instances of the SimplerImputer class

imputer = SimplerImputer(missing_values = np.nan, strategy = ‘mean’)

#Fit the imputer on the DataFrame 

imputer.fit(df)

#Apply the transform to the DataFrame 

df_imputed = imputer.transform(df)

#Print your updated matrix of features 

print(”Updated matrix of features: \n”, df_imputed)

-----------------------------

## HOT CODING
-when you turn code from words into numbers through repeating ones and zeros so the ML does not interpret them to have a relationship through order like one, two, three

### SPLITTING DATASET AND FEATURE SCALING 

#Import necessary libraries 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load the Iris dataset 

iris_df = pd.read_csv('iris.csv')

#Separate features and target 

X = iris_df.drop('target', axis = 1)
y = iris_df['target']

#split the dataset into an 80-20 training test set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Apply feature scaling on the training and test sets 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#print the scaled training and test sets 

print("Scaled Training Set:")
print(X_train)
print("\nScaled Test Set:")
print(X_test)

### FEATURE SCALING 
#Import necessary libraries 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load the Wine Quality dataset 

df = pd.read_csv('winequality-red.csv', delimiter=';')

#Separate features and target 

X = df.drop('quality', axis=1)
y = df['quality']

#Split the dataset into an 80-20 training-test set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Create an instance of the StandardScaler class 

sc = StandardScaler()

#Fit the Standard Scaler on the features from the training set and test set, transform it 

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Print the scaled training and test datasets 

print("Scaled training set:\n", X_train)
print("Scaled test set:\n", X_test)