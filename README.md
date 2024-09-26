# Password_Strength_Analyzer Using Markov Chains and n-grams with TF-IDF

## Overview
This project implements a machine learning model to classify password strength into three categories: **Weak**, **Normal**, and **Strong**. It leverages several techniques, including **Markov Chains**, **n-grams** with **TF-IDF vectorization**, and additional password features such as password length, lowercase, uppercase, digit, and special character frequencies. The password data comes from the publicly available **000webhost leak**.

The key stages of the project include:
- Data extraction and cleaning.
- Exploratory Data Analysis (EDA) to understand password patterns.
- Feature engineering, including Markov Chain likelihood and character frequencies.
- Model training using Logistic Regression, with hyperparameter tuning.
- Model evaluation and prediction.

## Project Structure
The project consists of the following components:
```
.
├── password_Data.sqlite               # SQLite database containing passwords and strength labels
├── markov_chain_model.pkl             # Trained Logistic Regression model
├── tfidf_vectorizer.pkl               # TF-IDF vectorizer used for n-grams
├── svd_transformer.pkl                # Truncated SVD object for dimensionality reduction
├── markov_transition_matrix.pkl       # Markov Chain transition matrix for password likelihood
├── password_strength_classifier.py    # Main Python script for model training, prediction, and saving components
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
```

## Prerequisites
Ensure you have Python 3.x installed, along with the following dependencies:
```
pip install pandas numpy matplotlib seaborn sklearn sqlite3
```
You can also install dependencies using the provided `requirements.txt` file:
```
pip install -r requirements.txt
```

## Dataset
The dataset is stored in password_Data.sqlite and contains two key columns:
- password: The actual password string.
- strength: The strength of the password, where 0 = Weak, 1 = Normal, and 2 = Strong.

## Steps and Methodology
1. **Data Extraction**

The password data is stored in an SQLite database, which is loaded into a pandas DataFrame using Python’s SQLite and pandas libraries.

2. **Data Cleaning**

We clean the data by removing irrelevant columns, duplicate rows, and checking for missing values.

3. **Exploratory Data Analysis (EDA)**

We analyze the structure of the passwords, looking at:
- Passwords with only numbers, uppercase letters, or special characters.
- Password length distribution and composition.

4. **Feature Engineering**

We create new features to better represent the characteristics of each password:
- Password length
- Frequency of lowercase, uppercase, digit, and special characters

5. **Markov Chain Model**

We use a Markov Chain to calculate the likelihood of character sequences in passwords. This helps measure how predictable a password is.

```
from collections import defaultdict

def build_transition_matrix(passwords):
    transition_matrix = defaultdict(lambda: defaultdict(int))
    for password in passwords:
        for i in range(len(password) - 1):
            char_1, char_2 = password[i], password[i + 1]
            transition_matrix[char_1][char_2] += 1
    for char_1 in transition_matrix:
        total = sum(transition_matrix[char_1].values())
        for char_2 in transition_matrix[char_1]:
            transition_matrix[char_1][char_2] /= total
    return transition_matrix

transition_matrix = build_transition_matrix(data["password"])
```

6. **n-grams with TF-IDF**

We apply TF-IDF vectorization to capture patterns of characters in passwords using n-grams (sequences of 2 to 4 characters).

7. **Dimensionality Reduction**

We reduce the dimensionality of the TF-IDF matrix using Truncated SVD to improve model efficiency.

8. **Model Training**

We train a Logistic Regression model on the combined features: Markov Chain likelihood, n-grams, password length, and character frequencies.

9. **Model Evaluation**

We evaluate the model using the test data and print out the classification report.

10. **Saving the Model**

Finally, we save the trained model, vectorizer, and other components using pickle for future use.
