# %% [markdown]
# <a href="https://colab.research.google.com/github/Adhirararaj/AutismPredictor/blob/main/Autism_Preidiction_using_machine_Learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# **1. Importing the dependencies**

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# %% [markdown]
# **2. Data Loading & Understanding**

# %%
# read the csv data to a pandas dataframe
df = pd.read_csv("/content/train.csv")

# %% [markdown]
# Initial Inspection

# %%
df.shape

# %%
df.head()

# %%
df.tail()

# %%
# display all columns of a dataframe
pd.set_option('display.max_columns', None)

# %%
df.info()

# %%
# convert age column datatype to integer
df["age"] = df["age"].astype(int)

# %%
df.head(2)

# %%
for col in df.columns:
  numerical_features = ["ID", "age", "result"]
  if col not in numerical_features:
    print(col, df[col].unique())
    print("-"*50)

# %%
# dropping ID & age_desc column
df = df.drop(columns=["ID", "age_desc"])

# %%
df.shape

# %%
df.head(2)

# %%
df.columns

# %%
df["contry_of_res"].unique()

# %%
# define the mapping dictionary for country names
mapping = {
    "Viet Nam": "Vietnam",
    "AmericanSamoa": "United States",
    "Hong Kong": "China"
}

# repalce value in the country column
df["contry_of_res"] = df["contry_of_res"].replace(mapping)

# %%
df["contry_of_res"].unique()

# %%
# taget class distribution
df["Class/ASD"].value_counts()

# %% [markdown]
# **Insights:**
# 1. missing values in ethnicity & relation
# 2. age_desc column has only 1 unique value. so it is removed as it is not important for prediction
# 3. fixed country names
# 4. identified class imbalance in the target column

# %% [markdown]
# **3. Exploratory Data Analysis (EDA)**

# %%
df.shape

# %%
df.columns

# %%
df.head(2)

# %%
df.describe()

# %% [markdown]
# **Univariate Analysis**

# %% [markdown]
# Numerical Columns:
# - age
# - result

# %%
# set the desired theme
sns.set_theme(style="darkgrid")

# %% [markdown]
# Distribution Plots

# %%
# Histogram for "age"

sns.histplot(df["age"], kde=True)
plt.title("Distribution of Age")

# calculate mean and median
age_mean = df["age"].mean()
age_median = df["age"].median()

print("Mean:", age_mean)
print("Median:", age_median)


# add vertical lines for mean and median
plt.axvline(age_mean, color="red", linestyle="--", label="Mean")
plt.axvline(age_median, color="green", linestyle="-", label="Median")

plt.legend()

plt.show()

# %%
# Histogram for "result"

sns.histplot(df["result"], kde=True)
plt.title("Distribution of result")

# calculate mean and median
result_mean = df["result"].mean()
result_median = df["result"].median()

print("Mean:", result_mean)
print("Median:", result_median)


# add vertical lines for mean and median
plt.axvline(result_mean, color="red", linestyle="--", label="Mean")
plt.axvline(result_median, color="green", linestyle="-", label="Median")

plt.legend()

plt.show()

# %% [markdown]
# **Box plots for identifying outliers in the numerical columns**

# %%
# box plot
sns.boxplot(x=df["age"])
plt.title("Box Plot for Age")
plt.xlabel("Age")
plt.show()

# %%
# box plot
sns.boxplot(x=df["result"])
plt.title("Box Plot for result")
plt.xlabel("result")
plt.show()

# %%
# count the outliers using IQR method
Q1 = df["age"].quantile(0.25)
Q3 = df["age"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
age_outliers = df[(df["age"] < lower_bound) | (df["age"] > upper_bound)]

# %%
len(age_outliers)

# %%
# count the outliers using IQR method
Q1 = df["result"].quantile(0.25)
Q3 = df["result"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
result_outliers = df[(df["result"] < lower_bound) | (df["result"] > upper_bound)]

# %%
len(result_outliers)

# %% [markdown]
# Univariate analysis of Categorical columns

# %%
df.columns

# %%
categorical_columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'gender',
       'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before',
       'relation']

for col in categorical_columns:
  sns.countplot(x=df[col])
  plt.title(f"Count Plot for {col}")
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.show()

# %%
# countplot for target column (Class/ASD)
sns.countplot(x=df["Class/ASD"])
plt.title("Count Plot for Class/ASD")
plt.xlabel("Class/ASD")
plt.ylabel("Count")
plt.show()

# %%
df["Class/ASD"].value_counts()

# %% [markdown]
# handle missing values in ethnicity and relation column

# %%
df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})

# %%
df["ethnicity"].unique()

# %%
df["relation"].unique()

# %%
df["relation"] = df["relation"].replace(
    {"?": "Others",
     "Relative": "Others",
     "Parent": "Others",
     "Health care professional": "Others"}
)

# %%
df["relation"].unique()

# %%
df.head()

# %% [markdown]
# **Label Encoding**

# %%
# identify columns with "object" data type
object_columns = df.select_dtypes(include=["object"]).columns

# %%
print(object_columns)

# %%
# initialize a dictionary to store the encoders
encoders = {}

# apply label encoding and store the encoders
for column in object_columns:
  label_encoder = LabelEncoder()
  df[column] = label_encoder.fit_transform(df[column])
  encoders[column] = label_encoder   # saving the encoder for this column


# save the encoders as a pickle file
with open("encoders.pkl", "wb") as f:
  pickle.dump(encoders, f)

# %%
encoders

# %%
df.head()

# %% [markdown]
# Bivariate Analysis

# %%
# correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

# %% [markdown]
# **Insights from EDA:**
# 
# - There are few outliers in the numerical columns (age, results)
# - There is a class imbalance in the target column
# - There is a class imbalance in the categorical features
# - We don't have any highly correlated column
# - performed label encoding and saved the encoders

# %% [markdown]
# **4. Data preprocessing**

# %% [markdown]
# Handling teh outliers

# %%
# function to replace the outliers with median
def replace_outliers_with_median(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  median = df[column].median()

  # replace outliers with median value
  df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)

  return df

# %%
# replace outliers in the "age" column
df = replace_outliers_with_median(df, "age")

# replace outliers in the "result" column
df = replace_outliers_with_median(df, "result")

# %%
df.head()

# %%
df.shape

# %% [markdown]
# **Train Test Split**

# %%
df.columns

# %%
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

# %%
print(X)

# %%
print(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
print(y_train.shape)
print(y_test.shape)

# %%
y_train.value_counts()

# %%
y_test.value_counts()

# %% [markdown]
# **SMOTE (Synthetic Minority Oversampling technique)**

# %%
smote = SMOTE(random_state=42)

# %%
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# %%
print(y_train_smote.shape)

# %%
print(y_train_smote.value_counts())

# %% [markdown]
# **5. Model Training**

# %%
# dictionary of classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# %%
# dictionary to store the cross validation results
cv_scores = {}

# perform 5-fold cross validation for each model
for model_name, model in models.items():
  print(f"Training {model_name} with default parameters...")
  scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
  cv_scores[model_name] = scores
  print(f"{model_name} Cross-Validation Accuracy: {np.mean(scores):.2f}")
  print("-"*50)

# %%
cv_scores

# %% [markdown]
# **6. Model Selection & Hyperparameter Tuning**

# %%
# Initializing models
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
xgboost_classifier = XGBClassifier(random_state=42)

# %%
# Hyperparameter grids for RandomizedSearchCV

param_grid_dt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20, 30, 50, 70],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}


param_grid_rf = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}


param_grid_xgb = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0]
}


# %%
# hyperparameter tunig for 3 tree based models

# the below steps can be automated by using a for loop or by using a pipeline

# perform RandomizedSearchCV for each model
random_search_dt = RandomizedSearchCV(estimator=decision_tree, param_distributions=param_grid_dt, n_iter=20, cv=5, scoring="accuracy", random_state=42)
random_search_rf = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid_rf, n_iter=20, cv=5, scoring="accuracy", random_state=42)
random_search_xgb = RandomizedSearchCV(estimator=xgboost_classifier, param_distributions=param_grid_xgb, n_iter=20, cv=5, scoring="accuracy", random_state=42)


# %%
# fit the models
random_search_dt.fit(X_train_smote, y_train_smote)
random_search_rf.fit(X_train_smote, y_train_smote)
random_search_xgb.fit(X_train_smote, y_train_smote)

# %%
# Get the model with best score

best_model = None
best_score = 0

if random_search_dt.best_score_ > best_score:
  best_model = random_search_dt.best_estimator_
  best_score = random_search_dt.best_score_

if random_search_rf.best_score_ > best_score:
  best_model = random_search_rf.best_estimator_
  best_score = random_search_rf.best_score_

if random_search_xgb.best_score_ > best_score:
  best_model = random_search_xgb.best_estimator_
  best_score = random_search_xgb.best_score_



# %%
print(f"Best Model: {best_model}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}")

# %%
# save the best model
with open("best_model.pkl", "wb") as f:
  pickle.dump(best_model, f)

# %% [markdown]
# **7. Evaluation**

# %%
# evaluate on test data
y_test_pred = best_model.predict(X_test)
print("Accuracy score:\n", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# %% [markdown]
# To do:
# 1. Build a Predictive system with encoders and model file
# 2. See if you could improve teh performance

# %%



