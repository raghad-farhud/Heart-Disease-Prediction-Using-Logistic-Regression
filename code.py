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
print(diseas_ds.TenYearCHD.value_counts())

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


# Fitting Logistic Regression Model for Heart Disease Prediction
from sklearn.linear_model import LogisticRegression

# initialize the model
logreg = LogisticRegression()
# fit the model with training data
logreg.fit(X_train, y_train)
# Make prediction on test data
y_pred = logreg.predict(X_test)


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



# To predict probability of having CHD in the next 10 years for a person:
# 1. Prepare person's data
person_data = [[56, 1, 0, 190, 140, 170]]

# 2. Normalize person's data
person_data_normalized = preprocessing.StandardScaler().fit(x).transform(person_data)

# 3. Predict CHD risk probability for the person
chd_probability = logreg.predict_proba(person_data_normalized)

# Extract the probability for CHD = 1 (positive class)
chd_probability_positive_class = chd_probability[0][1]

# Output the probability percentage
print("The person has a {:.2f}% probability of having a high risk of coronary heart disease (CHD) within the next 10 years.".format(chd_probability_positive_class * 100))
