# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns
# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# --- END OF IMPORT SECTION ---

# --- MAIN CODE ---
#importing the dataset
url = "Salary_Data.csv"
data = pd.read_csv(url)

#Visualizing the dataset
print(f"\nHere are the first 5 rows of the dataset:\n{data.head()}")

#Using a plot to visualize the data
plt.title("Years of Experience vs Salary") #title for the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(data["YearsExperience"], data["Salary"], color="blue") # actual plot
sns.regplot(data= data, x= "YearsExperience", y = "Salary") #regression line
plt.show() # renderize the plot to show it



# --- END OF MAIN CODE ---