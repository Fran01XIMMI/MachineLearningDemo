# --- IMPORT SECTION ---
import math
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
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)

#Visualizing the dataset
print(f"\nHere are the first 5 rows of the dataset:\n{data.head()}")

#Separate the data in features and target
x = data['YearsExperience'].values.reshape(-1,1)
y = data['Salary'].values.reshape(-1,1)

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
print(f"\n The total X size is: {x.shape[0]} and is the {x.shape[0]/x.shape[0] * 100} %")
print(f"\n The X train size is: {x_train.shape[0]} and is the {x_train.shape[0]/x.shape[0] * 100} % of the total X")
print(f"\n The X test size is: {x_test.shape[0]} and is the {x_test.shape[0]/x.shape[0] * 100} % of the total X")

print(f"\n The total Y size is: {y.shape[0]} and is the {y.shape[0]/y.shape[0] * 100} %")
print(f"\n The Y train size is: {y_train.shape[0]} and is the {y_train.shape[0]/y.shape[0] * 100} % of the total X")
print(f"\n The Y test size is: {y_test.shape[0]} and is the {y_test.shape[0]/y.shape[0] * 100} % of the total X")

#Visualizing data before scaling
print(f"\n-- BEFORE SCALING -- X_train:\n{x_train[:5]}")
print(f"\n-- BEFORE SCALING -- X_test:\n{x_test[:5]}")
print(f"\n-- BEFORE SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- BEFORE SCALING -- y_test:\n{y_test[:5]}")

#Feature scaling
scaler = StandardScaler()
# we are going to scale ONLY the features (i.e the X) and NOT the y!
x_train_scaled = scaler.fit_transform(x_train) #fitting to X_train and transforming them
x_test_scaled = scaler.transform(x_test) #transforming X_test. DO NOT FIT THEM!

#visualizing data after scaling
print(f"\n-- AFTER SCALING -- X_train:\n{x_train_scaled[:5]}")
print(f"\n-- AFTER SCALING -- X_test:\n{x_test_scaled[:5]}")
print(f"\n-- AFTER SCALING -- y_train:\n{y_train[:5]}")
print(f"\n-- AFTER SCALING -- y_test:\n{y_test[:5]}")

#linear Regression
model = LinearRegression()

#performing the training on the train data (i.e X_train_scaled, y_train)
model.fit(x_train_scaled, y_train)

#predicting new values
y_pred = model.predict(x_test_scaled)


#visualizing the parameters for the Regression after the training
print(f"\n After the training, the params for the Regressor are: {model.coef_}") #the coefficient of the model

# Visualizing the regression
plt.title("Years of Experience vs Salary") #title for the plot
plt.xlabel("Years of Experience") # title of x axis
plt.ylabel("Salary") # title of y axis
plt.scatter(x_test, y_test, color= 'red', label= 'Real Data') #actual plot
plt.plot(x_test,y_pred, color='blue', label= 'Predicted Data') #predicted data
plt.legend() #show the legend
plt.show()

#Evaluating the model
rmse = math.sqrt(mean_squared_error(y_test, y_pred)) # root mean squared error
print(f"\nThe Mean Squared Error is: {rmse:.2f}")
# --- END OF MAIN CODE ---

