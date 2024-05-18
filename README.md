                                                                                Description

This project implements a Neural Network for predicting the 'Age' attribute from a given dataset. The dataset is loaded from an Excel file, and the model uses ReLU as the activation function. The Neural Network architecture includes multiple hidden layers, and the training process is conducted using the backpropagation algorithm.

Key Features:
Data Loading:
- The dataset is loaded from an Excel file, specifically from the "Clear_Data" sheet.


Neural Network Architecture:
-The network is composed of an input layer, multiple hidden layers, and an output layer.
-ReLU (Rectified Linear Unit) activation function is used for the hidden layers.

Training:
-The model is trained using the backpropagation algorithm.
-The training process includes forward propagation, backpropagation, and weight/bias updates.
-The model is trained for 1000 epochs with a learning rate of 0.0001.

Prediction:
-After training, the model makes predictions on the same dataset.
-Predictions are compared with actual values to calculate accuracy.


Evaluation:
-The loss values during the training process are plotted to visualize the model's learning curve.
-The residuals (errors) between the actual and predicted values are analyzed through a histogram and a Q-Q plot to assess the model's performance.


Usage Instructions:
Install Required Libraries:

pip install numpy matplotlib pandas scipy openpyxl


Run the Project:

python Furkan_Atasert_Back_Propagation.py


Interpreting Results:
-The predicted values along with actual values are printed.
-The loss plot helps in understanding how well the model has learned over epochs.
-The histogram and Q-Q plot provide insights into the distribution and normality of prediction errors.

File Structure:
-Furkan_Atasert_Back_Propagation.py: Main script to run the project.
-simple-dataset.xlsx: Excel file containing the dataset.
-README.txt: Documentation and description of the project.

Developer:
-Furkan Atasert
