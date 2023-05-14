# Machine Learning Model - Diabetes Prediction<br>
This repository contains code for a machine learning model that predicts whether a person has diabetes or not based on various features.<br> The model is built using the Support Vector Machine (SVM) algorithm from the scikit-learn library.

##Dataset<br>
The model uses the "diabetes.csv" dataset, which contains information about pregnant women and their diabetes status. The dataset includes features such as pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.
<br><br>
##Files<br>
The main files in this repository are:
<br>
model.py: This file contains the implementation of the machine learning model. It includes steps to preprocess the data, split it into training and testing sets, train the SVM classifier, and make predictions.
<br>
diabetes.csv: This file is the dataset used for training and evaluating the model. It contains the input features and the corresponding diabetes outcome.
<br><br>
##Usage<br>
1.Install the required dependencies by running pip install numpy pandas sklearn.
2.Run the model.py file to train the model and make predictions. The code performs the following steps:
  *Load the dataset using pd.read_csv.
  *Preprocess the data by standardizing the input features using StandardScaler.
  *Split the data into training and testing sets using train_test_split.
  *Train the SVM classifier using the training data.
  *Evaluate the accuracy of the model on the training and testing data.
  *Make predictions on new input data and print the results.
3.Adjust the input data in the input_data variable to test the model's predictions on different samples.

Conclusion
This machine learning model demonstrates how SVM can be used to predict diabetes based on various health indicators. By training the model on the provided dataset, it achieves a reasonable accuracy in predicting whether a person has diabetes or not.
