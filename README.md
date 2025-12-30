# Salary Estimation using Artificial Neural Networks (ANN)

A machine learning project that predicts a bank customer's estimated salary based on their demographic and account information. This project features an ANN regression model built with TensorFlow/Keras and deployed as an interactive web application using Streamlit.

**🔴 Live Demo:** [Click here to use the App](https://ann-regression-churn-zabtskeggtqxq5udragvwp.streamlit.app/)

## 📌 Project Overview
The goal of this project is to build a regression model that can estimate a user's salary based on standard banking data points. It demonstrates the end-to-end process of data preprocessing, model training, and deployment.

**Key Features:**
* **Deep Learning Model:** An Artificial Neural Network (ANN) regressor trained on the Churn Modelling dataset.
* **Web Interface:** A user-friendly Streamlit app for real-time predictions.
* **Preprocessing Pipeline:** Automated handling of categorical data (Label Encoding & One-Hot Encoding) and feature scaling.

## 🛠️ Tech Stack
* **Python** (Programming Language)
* **TensorFlow & Keras** (Model Building)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-learn** (Data Preprocessing)
* **Streamlit** (Web Framework)
* **Pickle** (Model Persistence)

## 📂 Project Structure

```text
Salary-Estimation-Project/
│
├── venv/                          # Virtual environment (do not track in git)
│
├── README.md                      # Project documentation
├── requirements.txt               # List of python libraries needed
│
├── Churn_Modelling.csv            # The dataset
├── salaryregression.ipynb        # Notebook used to train the model
├── app.py                         # The Streamlit web application
│
├── # --- Generated Files (Created after running the notebook) ---
├── regression_model.h5                # The trained ANN model
├── salary_scaler.pkl              # Saved Standard Scaler
├── salary_gender_encoder.pkl      # Saved Label Encoder
└── salary_geo_encoder.pkl         # Saved One-Hot Encoder
```

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/salary-estimation-ann.git](https://github.com/yourusername/salary-estimation-ann.git)
    cd salary-estimation-ann
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install streamlit tensorflow pandas numpy scikit-learn
    ```

## 🏃‍♂️ How to Run

1.  **Train the Model (First time only):**
    Run the `salaryregression.ipynb` notebook to preprocess the data, train the ANN, and save the necessary `.h5` and `.pkl` files.

2.  **Start the Streamlit App:**
    ```bash
    streamlit run app2.py
    ```

3.  **Access the App:**
    Open your browser and go to `http://localhost:8501`.

## 📊 Model Architecture
The ANN consists of:
* **Input Layer:** Matches the number of preprocessed features.
* **Hidden Layers:** Two dense layers with ReLU activation (64 and 32 neurons).
* **Output Layer:** Single neuron with linear activation (for regression output).
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)

## 📝 Usage
Use the sidebar/main panel in the web app to input:
* Geography (France, Germany, Spain)
* Gender
* Age, Tenure, Balance
* Credit Score, Number of Products
* Credit Card Status & Active Membership

Click **"Estimate Salary"** to see the predicted value.
