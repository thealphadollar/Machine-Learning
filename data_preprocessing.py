import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# importing the dataset
data_set = pnd.read_csv('Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')
# print(data_set)
independent = data_set.iloc[:, :-1].values
# print(independent)
dependent = data_set.iloc[:, 3].values
# print(dependent)


# adding missing data
imputer = Imputer(missing_values="NaN", verbose=1, axis=0, copy=False)
imputer = imputer.fit(independent[:, 1:3])
independent[:, 1:3] = imputer.transform(independent[:, 1:3])
# print(independent)


# labeling textual data
labelencoder_independent = LabelEncoder()
labelencoder_independent.fit(independent[:, 0])
independent[:, 0] = labelencoder_independent.transform(independent[:, 0])
# print(independent)


# categorizing the data into columns
hotencorder_independent = OneHotEncoder(categorical_features=[0])
hotencorder_independent.fit(independent)
independent = hotencorder_independent.transform(independent).toarray()
# print(independent)


labelencoder_dependent = LabelEncoder()
labelencoder_dependent.fit(dependent[:])
dependent[:] = labelencoder_dependent.transform(dependent[:])
# print(dependent)


# spilting data into training set and testing set
independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent,
                                                                                        dependent, test_size=.2,
                                                                                        random_state=0)
# print(independent_train)
# print(independent_test)


# scaling data
std_scaler = StandardScaler()
std_scaler.fit(independent_train[:, 3:])
independent_train[:, 3:] = std_scaler.transform(independent_train[:, 3:])
independent_test[:, 3:] = std_scaler.transform(independent_test[:, 3:])
# print(independent_train)
# print(independent_test)


