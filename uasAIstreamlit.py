import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('data.csv')

# inspecting the first 5 rows of the dataframe
st.write(car_dataset.head())

# checking the number of rows and columns
st.write(car_dataset.shape)

# getting some information about the dataset
st.write(car_dataset.info())

# checking the number of missing values
st.write(car_dataset.isnull().sum())

# checking the distribution of categorical data
st.write(car_dataset.Fuel_Type.value_counts())
st.write(car_dataset.Seller_Type.value_counts())
st.write(car_dataset.Transmission.value_counts())

# Encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}}, inplace=True)

# Encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}}, inplace=True)

# Encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}}, inplace=True)

st.write(car_dataset.head())

# Splitting the data and Target
X = car_dataset.drop(['Car_Name','Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

st.write(X)
st.write(Y)

# Splitting Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Model Training - Linear Regression

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

# Model Evaluation

training_data_prediction = lin_reg_model.predict(X_train)

error_score = metrics.r2_score(Y_train, training_data_prediction)
st.write("R squared Error (Train): ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Train)")
st.pyplot()

test_data_prediction = lin_reg_model.predict(X_test)

error_score = metrics.r2_score(Y_test, test_data_prediction)
st.write("R squared Error (Test): ", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Test)")
st.pyplot()

# Model Training - Lasso Regression

lass_reg_model = Lasso()
lass_reg_model.fit(X_train,Y_train)

training_data_prediction = lass_reg_model.predict(X_train)

error_score = metrics.r2_score(Y_train, training_data_prediction)
st.write("R squared Error (Train): ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Train)")
st.pyplot()

test_data_prediction = lass_reg_model.predict(X_test)

error_score = metrics.r2_score(Y_test, test_data_prediction)
st.write("R squared Error (Test): ", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices (Test)")
st.pyplot()
