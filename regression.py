# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
import seaborn as sns
# importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split #to perform train-test split
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

#Separate the data in features and target
x = data['YearsExperience']
y = data['Salary']

#Using a plot to visualize the data
plt.title("Years of Experience vs Salary") #title for the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(x,y, color= 'red') #actual plot
plt.scatter(data["YearsExperience"], data["Salary"], color="blue") # actual plot
sns.regplot(data= data, x= "YearsExperience", y = "Salary") #regression line
plt.show() # renderize the plot to show it

#Splitting the dataset into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101)

#Checking the train and test size to prove they are 80% and 20% respectively
print(f"\n The total X size is: {x.shape[0]}")
print(f"\n The X train size is: {x_train.shape[0]}")
print(f"\n The X test size is: {x_test.shape[0]}")

# --- END OF MAIN CODE ---