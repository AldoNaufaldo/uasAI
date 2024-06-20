import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

# Fungsi untuk memuat data dan memprosesnya
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        st.error(f"Tidak dapat menemukan file: {filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error dalam memuat data: {str(e)}")
        st.stop()

# Fungsi untuk membuat scatter plot dan mengembalikan figur
def plot_scatter(x, y, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Harga yang sebenarnya")
    ax.set_ylabel("Harga yang diprediksi")
    ax.set_title(title)
    return fig

# Fungsi untuk menampilkan data dalam tabel di bawah plot
def display_data_table(data, title):
    st.write(f"### {title}")
    st.dataframe(data)

# Aplikasi utama Streamlit
def main():
    st.title("Prediksi Harga Mobil")

    # Sidebar untuk mengunggah file
    st.sidebar.title("Unggah File")
    training_file = st.sidebar.file_uploader("Unggah Data Latihan (CSV)", type=['csv'])
    testing_file = st.sidebar.file_uploader("Unggah Data Uji (CSV)", type=['csv'])

    if training_file and testing_file:
        st.sidebar.info("File berhasil diunggah")

        # Memuat data latihan dan uji
        train_data = load_data(training_file)
        test_data = load_data(testing_file)

        # Menampilkan data dan info dasar
        st.write("### Dataframe Data Latihan:")
        st.dataframe(train_data)
        st.write("### Bentuk DataFrame:")
        st.write(train_data.shape)
        st.write("### Informasi Data Latihan:")
        st.write(train_data.info())

        st.write("### Dataframe Data Uji:")
        st.dataframe(test_data)
        st.write("### Bentuk DataFrame:")
        st.write(test_data.shape)
        st.write("### Informasi Data Uji:")
        st.write(test_data.info())

        # Encoding kolom kategorikal
        train_data = pd.get_dummies(train_data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)
        test_data = pd.get_dummies(test_data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

        # Memisahkan data dan target
        X_train = train_data.drop(['Car_Name', 'Selling_Price'], axis=1)
        Y_train = train_data['Selling_Price']

        X_test = test_data.drop(['Car_Name', 'Selling_Price'], axis=1)
        Y_test = test_data['Selling_Price']

        # Pelatihan Model dan Evaluasi - Regresi Linear
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, Y_train)

        training_data_prediction = lin_reg_model.predict(X_train)
        train_error = metrics.r2_score(Y_train, training_data_prediction)

        st.write("### Grafik Hasil Latihan dengan Regresi Linear:")
        fig_train = plot_scatter(Y_train, training_data_prediction, "Harga yang sebenarnya vs Harga yang diprediksi (Latihan)")
        st.pyplot(fig_train)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_train, 'Harga yang diprediksi': training_data_prediction}), "Hasil Latihan dengan Regresi Linear")

        test_data_prediction = lin_reg_model.predict(X_test)
        test_error = metrics.r2_score(Y_test, test_data_prediction)

        st.write("### Grafik Hasil Uji:")
        fig_test = plot_scatter(Y_test, test_data_prediction, "Harga yang sebenarnya vs Harga yang diprediksi")
        st.pyplot(fig_test)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_test, 'Harga yang diprediksi': test_data_prediction}), "Hasil Uji")

        # Pelatihan Model dan Evaluasi - Lasso Regression
        lasso_reg_model = Lasso()
        lasso_reg_model.fit(X_train, Y_train)

        training_data_prediction_lasso = lasso_reg_model.predict(X_train)
        train_error_lasso = metrics.r2_score(Y_train, training_data_prediction_lasso)

        st.write("### Grafik Hasil Latihan dengan Lasso Regression:")
        fig_train_lasso = plot_scatter(Y_train, training_data_prediction_lasso, "Harga yang sebenarnya vs Harga yang diprediksi (Latihan)")
        st.pyplot(fig_train_lasso)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_train, 'Harga yang diprediksi': training_data_prediction_lasso}), "Hasil Latihan dengan Lasso Regression")

        test_data_prediction_lasso = lasso_reg_model.predict(X_test)
        test_error_lasso = metrics.r2_score(Y_test, test_data_prediction_lasso)

        st.write("### Grafik Hasil Uji:")
        fig_test_lasso = plot_scatter(Y_test, test_data_prediction_lasso, "Harga yang sebenarnya vs Harga yang diprediksi")
        st.pyplot(fig_test_lasso)
        display_data_table(pd.DataFrame({'Harga yang sebenarnya': Y_test, 'Harga yang diprediksi': test_data_prediction_lasso}), "Hasil Uji")

        # Menutup figur untuk membebaskan sumber daya (opsional)
        plt.close(fig_train)
        plt.close(fig_test)
        plt.close(fig_train_lasso)
        plt.close(fig_test_lasso)

    else:
        st.info("Silakan unggah file CSV.")

if __name__ == '__main__':
    main()
