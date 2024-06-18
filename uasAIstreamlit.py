import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Function to load data and process it
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        st.error(f"Could not find the file: {filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Function to plot scatter plot and return the figure
def plot_scatter(x, y, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(title)
    return fig

# Function to display data in a table below the plot
def display_data_table(data, title):
    st.write(f"### {title} Data:")
    st.write(data)

# Main Streamlit app
def main():
    st.title("Car Price Prediction")

    # Sidebar for uploading files
    st.sidebar.title("Upload Files")
    training_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=['csv'])
    testing_file = st.sidebar.file_uploader("Upload Testing Data (CSV)", type=['csv'])

    if training_file and testing_file:
        st.sidebar.info("Files uploaded successfully!")

        # Load training and testing data
        train_data = load_data(training_file)
        test_data = load_data(testing_file)

        # Display data and basic info
        st.write("### Training Data:")
        st.write(train_data.head())
        st.write(train_data.shape)
        st.write(train_data.info())
        st.write(train_data.isnull().sum())

        st.write("### Testing Data:")
        st.write(test_data.head())
        st.write(test_data.shape)
        st.write(test_data.info())
        st.write(test_data.isnull().sum())

        # Encoding categorical columns
        categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), categorical_cols)
            ], remainder='passthrough'
        )

        # Define pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())  # You can change this to Lasso() if needed
        ])

        # Splitting the data and Target
        X_train = train_data.drop(['Car_Name', 'Selling_Price'], axis=1)
        Y_train = train_data['Selling_Price']

        X_test = test_data.drop(['Car_Name', 'Selling_Price'], axis=1)
        Y_test = test_data['Selling_Price']

        # Model Training and Evaluation
        pipeline.fit(X_train, Y_train)

        # Predictions
        training_data_prediction = pipeline.predict(X_train)
        train_error = metrics.r2_score(Y_train, training_data_prediction)

        fig_train = plot_scatter(Y_train, training_data_prediction, "Actual Prices vs Predicted Prices (Train)")
        st.pyplot(fig_train)
        display_data_table(pd.DataFrame({'Actual Price': Y_train, 'Predicted Price': training_data_prediction}), "Training")

        test_data_prediction = pipeline.predict(X_test)
        test_error = metrics.r2_score(Y_test, test_data_prediction)

        fig_test = plot_scatter(Y_test, test_data_prediction, "Actual Prices vs Predicted Prices (Test)")
        st.pyplot(fig_test)
        display_data_table(pd.DataFrame({'Actual Price': Y_test, 'Predicted Price': test_data_prediction}), "Test")

        # Closing figures to release resources (optional)
        plt.close(fig_train)
        plt.close(fig_test)

    else:
        st.info("Please upload both training and testing CSV files in the sidebar.")

if __name__ == '__main__':
    main()
