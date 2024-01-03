# Diabetes Prediction using Naive Bayes Classifier

This repository contains code for predicting diabetes using the Naive Bayes classifier. The model is built using scikit-learn's Gaussian Naive Bayes implementation and trained on the Diabetes Prediction dataset.

## Importing Libraries and Dataset
The code starts by importing necessary libraries such as numpy, pandas, and scikit-learn. It reads the dataset `diabetes_prediction_dataset.csv` using Pandas.

## Data Preprocessing
The dataset is split into target ('diabetes') and features ('X'). Preprocessing steps include handling categorical variables ('gender' and 'smoking_history') using dummy variables and one-hot encoding.

## Training and Testing
The dataset is split into training and testing sets using a 80:20 ratio. The Gaussian Naive Bayes classifier (`GaussianNB()`) is initialized, trained on the training data, and used to make predictions (`y_pred`) on the test set.

## Evaluation
The accuracy of the model is evaluated using a custom accuracy function, and the result is printed to the console.

## Usage
To use this code:
1. Clone the repository:

    ```
    git clone https://github.com/Mukund604/Diabetes-Prediction.git
    ```

2. Ensure you have Python installed along with the necessary libraries mentioned in the code.

3. Run the script `diabetes_prediction.py` or Jupyter Notebook containing this code.

## Files
- `diabetes_prediction.py`: Python script containing the code for diabetes prediction.
- `diabetes_prediction_dataset.csv`: Dataset used for diabetes prediction.

## Results
The accuracy achieved by the Naive Bayes model on the test set is printed to the console.

## License
This project is not licensed.

