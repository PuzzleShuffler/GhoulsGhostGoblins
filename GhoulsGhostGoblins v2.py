##### Load Modules #####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

##### Load dataset #####
original_Data = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

##### Summarize Data #####
# Shape of Data Set
print('The dataset has %d rows and %d columns.' % (original_Data.shape[0], original_Data.shape[1]))
print("")

# Information Columns
original_Data.info()
print("")

# Data Types of Columns
original_Data.dtypes
print("")

# Get Categorical Columns
cat_col = list(original_Data.drop(['id'], axis='columns').select_dtypes(include=['object', 'bool']).columns)

# Class Distrubtion of Categorical Columns
for col in cat_col:
    print('Class Distribution of %s is:' % (col))
    print(original_Data.groupby([col]).size())
    print("")

# Get Numeric Columns
num_col = list(original_Data.drop(['id'], axis='columns').select_dtypes(include=['int64', 'float64']).columns)

# Distribution of Numeric Columns
for col in num_col:
    print('Distribution of %s is:' % (col))
    print(original_Data[col].describe())
    print("")

# Describe all Data
original_Data.describe(include='all')

# Correlation of Numeric Columns
original_Data[num_col].corr()

# Skewness of Numeric Columns
original_Data.iloc[:, 1:].skew()

##### Data visualizations #####
# Histograms of Numeric Columns
for col in num_col:
    sns.histplot(original_Data[col], kde=True)
    plt.show()
print("")

# Count Plots of Categorical Columns
for col in cat_col:
    sns.countplot(original_Data[col])
    plt.show()
print("")

# Correlation Heat Map of Numeric Columns
sns.heatmap(original_Data[num_col].corr(), cmap="coolwarm", annot=True)
plt.show()
print("")

# Pairplot of Numeric Columns
sns.pairplot(original_Data[num_col])
plt.show()
print("")

##### Data Cleaning #####
# Number of unique values in each column
print('Columns and their number of unique values:')
print(original_Data.drop(['id'], axis='columns').nunique())
print("")

# Outliers
lof = LocalOutlierFactor()
yhat = lof.fit_predict(original_Data[num_col])

mask = yhat == -1 # mask for outliers, 1 is normal, -1 is outlier
print('There are %d outliers according to LocalOutlierFactor.' % len(original_Data[mask]))
print(original_Data[mask]) # Outliers
print("")

mask = yhat == 1  # mask for normal, 1 is normal, -1 is outlier
outlierFreeData = original_Data[mask]

# Missing Values
print('There are %d missing pieces of data.' % outlierFreeData.isna().sum().sum())
outlierFreeData.isna().sum()
print("")

##### Evaluate Algorithms #####
# Polynomial Features of Degree 4
t=[('num', PolynomialFeatures(degree=4), num_col)]
col_transform=ColumnTransformer(transformers=t, remainder='passthrough')

# define Logistic Regression Model
model=LogisticRegression()

# Pipeline Steps
pipeline=Pipeline(steps=[('t', col_transform), ('m', model)])

# Repeated CV
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

# X Data
X=outlierFreeData.drop(['id', 'type', 'color'], axis='columns')

# Y Data
y=outlierFreeData[['type']]

# Score Model
scores=cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# Mean Accuracy
print('Accuracy of %s:' % (model.__module__))
print("%f +/- %f" % (scores.mean(), scores.std()))

# Fit Pipeline on Train
pipeline.fit(X,y)

# Predict on Test Data
pred = pipeline.predict(testData)

# Submission
submission = pd.DataFrame({'id':testData['id'], 'type':pred})
submission.to_csv('submissions/logistic regression - poly 4.csv', index=False)
