import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Loading the data from csv file to pandas dataframe
try:
    car_dataset = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("Could not find the data file.")
    st.stop()

# Displaying data and basic info
st.write(car_dataset.head())
st.write(car_dataset.shape)
st.write(car_dataset.info())
st.write(car_dataset.isnull().sum())

# Encoding categorical columns
car_dataset = pd.get_dummies(car_dataset, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Splitting the data and Target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Splitting Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Function to plot scatter plot and return the figure
def plot_scatter(x, y, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
    return fig

# Model Training and Evaluation - Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

training_data_prediction = lin_reg_model.predict(X_train)
train_error = metrics.r2_score(Y_train, training_data_prediction)

fig_train = plot_scatter(Y_train, training_data_prediction, "Actual Prices vs Predicted Prices (Train)")
st.pyplot(fig_train)

test_data_prediction = lin_reg_model.predict(X_test)
test_error = metrics.r2_score(Y_test, test_data_prediction)

fig_test = plot_scatter(Y_test, test_data_prediction, "Actual Prices vs Predicted Prices (Test)")
st.pyplot(fig_test)

# Model Training and Evaluation - Lasso Regression
lasso_reg_model = Lasso()
lasso_reg_model.fit(X_train, Y_train)

training_data_prediction_lasso = lasso_reg_model.predict(X_train)
train_error_lasso = metrics.r2_score(Y_train, training_data_prediction_lasso)

fig_train_lasso = plot_scatter(Y_train, training_data_prediction_lasso, "Actual Prices vs Predicted Prices (Train) - Lasso")
st.pyplot(fig_train_lasso)

test_data_prediction_lasso = lasso_reg_model.predict(X_test)
test_error_lasso = metrics.r2_score(Y_test, test_data_prediction_lasso)

fig_test_lasso = plot_scatter(Y_test, test_data_prediction_lasso, "Actual Prices vs Predicted Prices (Test) - Lasso")
st.pyplot(fig_test_lasso)

# Closing figures to release resources (optional)
plt.close(fig_train)
plt.close(fig_test)
plt.close(fig_train_lasso)
plt.close(fig_test_lasso)
