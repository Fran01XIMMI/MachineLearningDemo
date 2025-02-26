import numpy as np
import pandas as pd # for DataFrames
import matplotlib.pyplot as plt # to visualise the data
from sklearn.model_selection import train_test_split # to split the data into training and testing sets
from sklearn.preprocessing import StandardScaler # to standardize (scale) the features (the X)
from sklearn.ensemble import RandomForestClassifier # to use the RandomForest model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # to evaluate the model
import seaborn as sns # to visualize the confusion matrix
from sklearn.datasets import load_iris #to load the Iris dataset
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
#importing the dataset: the Iris dataset contains data of three species of flowers
dataset = load_iris()

# Creating the DataFrame
data = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
data['target'] = dataset.target #target aka the y

#visualizing the first row of the dataset
print(f"\nHere are the first 5 rows of the dataset:\n {data.head()}")


# --- END OF MAIN CODE ---