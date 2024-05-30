# Heart-Disease-Prediction-Using-Logistic-Regression
World Health Organization has estimated that four out of five cardiovascular disease (CVD) deaths are due to heart attacks. This whole research intends to pinpoint the ratio of patients who possess a good chance of being affected by CVD and also to predict the overall risk using Logistic Regression.


### #1 install and import the following librares:

- **pandas**: Data manipulation and analysis.
- **pylab**: Plotting and numerical operations.
- **numpy**: Numerical computing and array operations.
- **scipy.optimize**: Mathematical optimization.
- **statsmodels**: Statistical modeling and hypothesis testing.
- **sklearn.preprocessing**: Data preprocessing.
- **matplotlib.pyplot**: Plotting and visualizations.
- **matplotlib.mlab**: Deprecated, array functions (not used in tutorial).
- **seaborn**: Statistical data visualization.

```python
# pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn

# Importing necessary libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
```

### **Data Preparation**

The [**dataset**](https://media.geeksforgeeks.org/wp-content/uploads/20240307152534/framingham.csv) is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD). The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.

```python

diseas_ds = pd.read_csv("./framingham.csv")

# drop unrelevent attribute
diseas_ds.drop(["education"], inplace=True, axis=1) 
# rename column for clarity
diseas_ds.rename(columns={'male': 'Is_male'}, inplace=True)
# drop rows that have missing values such as Null
diseas_ds.dropna(axis=0, inplace=True)

print(diseas_ds.head())
print(diseas_ds.shape)
# The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD)
# Counting the Values of the Target Variable
print(diseas_ds['TenYearCHD'].value_counts)
```

![Heart Disease Prediction Using Logistic Regression -1](https://github.com/raghad-farhud/Heart-Disease-Prediction-Using-Logistic-Regression/assets/86526536/60a8715d-2183-404d-9df0-40459a8f094b)
![Heart Disease Prediction Using Logistic Regression -2](https://github.com/raghad-farhud/Heart-Disease-Prediction-Using-Logistic-Regression/assets/86526536/16889425-4b86-4bbc-a8cf-3b14b2eb42c6)


### **Next Step: Splitting the Dataset into Training and Testing Sets**

```python
# Selecting features and target variables
x = np.asarray(diseas_ds[['age', 'Is_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(diseas_ds['TenYearCHD'])

# Normalizing the dataset
# `preprocessing.StandardScaler()`: Scales the features such that they have a mean of 0 and a standard deviation of 1.
# `fit(X)`: Computes the mean and standard deviation for scaling.
# `transform(X)`: Applies the scaling transformation to the feature data.
X = preprocessing.StandardScaler().fit(x).transform(x)

# Splitting Dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

print("Train set: ", X_train.shape, y_train.shape)
print("Test set: ", X_test.shape, y_test.shape)

```

![Heart Disease Prediction Using Logistic Regression -3](https://github.com/raghad-farhud/Heart-Disease-Prediction-Using-Logistic-Regression/assets/86526536/9f121363-d28d-4c4f-8ff7-c869406dd192)


### **Exploratory Data Analysis (EDA)**

We'll use visualization to explore the dataset and gain insights. Let's start with visualizing the distribution of the target variable **`TenYearCHD`** and then move on to other visualizations to understand the relationships between features and the target variable.

```python
# Explarotary Data Analysis EDA
# Counting the Number of Patients Affected by CHD:
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=diseas_ds, palette="BuGn_r", hue='TenYearCHD', dodge=False, legend=False)
plt.title('Distribution of TenYearCHD')
plt.xlabel('TenYearCHD (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('./images/distribution_tenyearchd.png')

# Distribution of Age:
plt.figure(figsize=(10, 6))
sns.histplot(diseas_ds['age'], kde=True, bins=30, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('./images/distribution_age.png')

# Correlation Heatmap:
plt.figure(figsize=(12, 8))
correlation_matrix = diseas_ds.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.2)
plt.title('Correlation Heatmap')
plt.savefig('./images/correlation_heatmap.png')
```

![Heart Disease Prediction Using Logistic Regression -4](https://github.com/raghad-farhud/Heart-Disease-Prediction-Using-Logistic-Regression/assets/86526536/d7a022a1-8d15-4814-8714-fb2cfabacb39)
![Heart Disease Prediction Using Logistic Regression -5](https://github.com/raghad-farhud/Heart-Disease-Prediction-Using-Logistic-Regression/assets/86526536/7ef0aced-8d95-4c22-97ca-f525f6440068)
![Heart Disease Prediction Using Logistic Regression -6](https://github.com/raghad-farhud/Heart-Disease-Prediction-Using-Logistic-Regression/assets/86526536/d08f4335-57f6-4adf-acea-0d9cefe94e61)


### **Fitting Logistic Regression Model for Heart Disease Prediction**

```python
# Fitting Logistic Regression Model for Heart Disease Prediction

from sklearn.linear_model import LogisticRegression

# initialize the model
logreg = LogisticRegression()
# fit the model with training data
logreg.fit(X_train, y_train)
# Make prediction on test data
y_pred = logreg.predict(X_test)
```

### **Evaluating Logistic Regression Model**

```python
# Evaluating Logistic Regression Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Evaluate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of the model:', accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Generate classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)
```

### **Explanation:**

- **Accuracy**: The proportion of correctly classified samples out of the total number of samples. In this case, the accuracy is approximately 0.849 or 84.9%.
- **Confusion Matrix**:
    - True Negative (TN): 942
    - False Positive (FP): 9
    - False Negative (FN): 161
    - True Positive (TP): 14
    - The confusion matrix shows how many samples were correctly and incorrectly classified by the model.
- **Classification Report**:
    - **Precision**: The proportion of correctly predicted positive cases out of all cases predicted as positive. Precision for class 0 (no risk of heart disease) is 0.85, and for class 1 (risk of heart disease) is 0.61.
    - **Recall**: The proportion of correctly predicted positive cases out of all actual positive cases. Recall for class 0 is 0.99, and for class 1 is 0.08.
    - **F1-score**: The harmonic mean of precision and recall. It provides a balance between precision and recall. F1-score for class 0 is 0.92, and for class 1 is 0.14.
    - **Support**: The number of actual occurrences of each class in the test data.

### To predict probability of having CHD in the next 10 years for a person:

```python
# To predict probability of having CHD in the next 10 years for a person:

# Data sample:
#     'age': 56,
#     'Is_male': 1,  # Assuming 1 for male, 0 for female
#     'cigsPerDay': 0,
#     'totChol': 190,
#     'sysBP': 140,
#     'glucose': 120

# 1. Prepare person's data
person_data = [[56, 1, 0, 190, 140, 120]]

# 2. Normalize person's data
person_data_normalized = preprocessing.StandardScaler().fit(x).transform(person_data)

# 3. Predict CHD risk probability for the person
chd_probability = logreg.predict_proba(person_data_normalized)

# Extract the probability for CHD = 1 (positive class)
chd_probability_positive_class = chd_probability[0][1]

# Output the probability percentage
print("The person has a {:.2f}% probability of having a high risk of coronary heart disease (CHD) within the next 10 years.".format(chd_probability_positive_class * 100))
```

**Output:**

The person has a 24.96% probability of having a high risk of coronary heart disease (CHD) within the next 10 years.
