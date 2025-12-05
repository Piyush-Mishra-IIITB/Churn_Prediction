Customer Churn Prediction using Artificial Neural Network (ANN)

This project implements a Customer Churn Prediction System using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The application is deployed using Streamlit, providing an interactive UI where users can input customer details and receive a churn prediction.

The model is trained on a bank churn dataset and predicts whether a customer is likely to leave the bank based on features such as credit score, age, tenure, balance, geography, and activity status.

📌 Features

Deep Learning–based classifier using ANN

Data preprocessing with:

One-hot encoding (Geography, Gender)

Feature scaling using StandardScaler

Interactive UI built with Streamlit

Real-time prediction based on user input

Clean and professional interface with improved design layout

Probabilistic churn output with interpretation

🧠 Model Architecture

The ANN consists of:

Input layer with 11 features

Two dense hidden layers with ReLU activation

Output layer with Sigmoid activation (binary classification)

The model predicts:

1 → Customer will churn

0 → Customer will not churn