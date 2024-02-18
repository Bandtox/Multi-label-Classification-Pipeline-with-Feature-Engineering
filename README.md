# Multi-label Classification Project

This project focuses on building a multi-label classification model to predict multiple labels for a given set of input features. The dataset used for this project consists of various features and multiple target labels.

## Overview

The project involves several key steps:

1. **Data Importing**: Importing necessary libraries and loading the dataset from CSV files using pandas.
2. **Data Preprocessing**:
   - Merging DataFrames: Combining training data and labels into a single DataFrame.
   - Handling Missing Values: Using SimpleImputer from scikit-learn to impute missing values in both numerical and categorical columns.
   - Scaling Numerical Data: Standardizing numerical features to ensure they have the same scale.
   - Feature Hashing: Reducing the dimensionality of categorical features using FeatureHasher.
   - Encoding Categorical Features: Label encoding categorical features for model compatibility.
   - Separating Features and Labels: Splitting the dataset into feature matrix and target vector.
3. **Model Training**: Training a Decision Tree Classifier on the preprocessed data.
4. **Model Evaluation**: Evaluating the performance of the trained model using accuracy and confusion matrix.
5. **Test Data Processing**:
   - Loading Test Dataset: Loading and preprocessing the test dataset similar to the training data.
   - Feature Hashing and Encoding: Applying feature hashing and label encoding to the test data.
   - Data Consistency Checks: Ensuring consistency in column order between training and test datasets.
6. **Model Prediction**: Making predictions on the test data using the trained model.
7. **Output Generation**: Generating output predictions in a suitable format for submission.
8. **Saving Model and Output**: Saving the trained model and output predictions to files for future use and submission.

## Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## Dataset

The dataset used in this project contains various features (both numerical and categorical) and multiple target labels. The goal is to predict the labels based on the input features. The dataset is split into training and test sets for model development and evaluation.

## How to Use

1. Clone the repository to your local machine.
2. Ensure you have Python and the required libraries installed.
3. Run the provided code cells or scripts to preprocess the data, train the model, make predictions, and generate output.
4. Customize the code or parameters as needed for your specific use case.
5. Explore the results, evaluate model performance, and make improvements as necessary.
