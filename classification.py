import numpy as np
import pandas as pd # for DataFrames
import matplotlib.pyplot as plt # to visualise the data
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.preprocessing import StandardScaler # to standardize (scale) the features (the X)
from sklearn.ensemble import RandomForestClassifier # to use the RandomForest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # to evaluate the model
import seaborn as sns # to visualize the confusion matrix
from sklearn.datasets import load_iris #to load the Iris dataset

# from regression import x_train, x_test, y_train, y_test, y_pred

# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
#importing the dataset: the Iris dataset contains data of three species of flowers
dataset = load_iris()

# Creating the DataFrame
data = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
data['target'] = dataset.target #target aka the y

#visualizing the first row of the dataset
print(f"\nHere are the first 5 rows of the dataset:\n {data.head()}")

#seperating the data in features and target
x = data.iloc[:, :-1].values # all the columns except the last one
y = data['target'].values # the last column

#splitting the dataset into training and test
x_train, x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=101, stratify=y)
# note: the 'stratify' parameter that classes are wellbalanced between train and test

#Feature scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train) #fitting to X_train and transforming them
x_test_scaled = scaler.transform(x_test) #transforming X_test. DO NOT FIT THEM!

# Creating the model
model = RandomForestClassifier(n_estimators=100, random_state=101)

#Training model
model.fit(x_train_scaled, y_train)

#prediction over the test
y_pred = model.predict(x_test_scaled)

# --- END OF MAIN CODE ---