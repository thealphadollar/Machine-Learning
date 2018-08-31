import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer


# importing the dataset
data_set = pnd.read_csv('Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')
# print(data_set)
independent = data_set.iloc[:, :-1].values
# print(independent)
dependent = data_set.iloc[:, 3].values
# print(dependent)

imputer = Imputer(missing_values="NaN", verbose=1, axis=0, copy=False)
imputer = imputer.fit(independent[:, 1:3])
independent[:, 1:3] = imputer.transform(independent[:, 1:3])

print(independent)
