# Generated from: 8. Logistic Regression(1).ipynb
# Converted at: 2026-04-27T11:23:00.939Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ToyotaCorolla - MLR.csv")

# Clean column names (removes extra spaces, fixes case issues)
df.columns = df.columns.str.strip()

# Display basic information
print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nDataset Info:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# --------- EDA VISUALIZATION ---------

# Distribution of target variable (Price)
if "Price" in df.columns:
    plt.figure()
    sns.histplot(df["Price"], kde=True)
    plt.title("Distribution of Car Price")
    plt.show()

    # Price vs Age
    plt.figure()
    sns.scatterplot(x=df["Age_08_04"], y=df["Price"])
    plt.title("Price vs Age")
    plt.show()
else:
    print("\n 'Price' column not found. Please check target column name.")

# Correlation heatmap (only numeric columns)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Categorical vs Price (if Fuel_Type exists)
if "Fuel_Type" in df.columns and "Price" in df.columns:
    plt.figure()
    sns.boxplot(x="Fuel_Type", y="Price", data=df)
    plt.title("Price vs Fuel Type")
    plt.show()


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("ToyotaCorolla - MLR.csv")

# Clean column names
df.columns = df.columns.str.strip()

# --------- FEATURE EXAMINATION ---------

# 1. List of features
print("Features in the dataset:\n")
print(df.columns.tolist())

# 2. Data types of features
print("\nData Types of Features:\n")
print(df.dtypes)

# 3. Separate numerical and categorical features
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical Features:\n", numerical_features)
print("\nCategorical Features:\n", categorical_features)

# 4. Summary statistics for numerical features
print("\nSummary Statistics (Numerical Features):\n")
print(df[numerical_features].describe())

# 5. Summary statistics for categorical features
if len(categorical_features) > 0:
    print("\nSummary Statistics (Categorical Features):\n")
    print(df[categorical_features].describe())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ToyotaCorolla - MLR.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ---------- HISTOGRAMS (Distribution of Numerical Features) ----------
numerical_features = df.select_dtypes(include=np.number).columns

df[numerical_features].hist(figsize=(14,10), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# ---------- BOX PLOTS (Outlier Detection) ----------
plt.figure(figsize=(14,6))
sns.boxplot(data=df[numerical_features])
plt.xticks(rotation=90)
plt.title("Box Plot of Numerical Features")
plt.show()

# ---------- PAIR PLOT (Feature Relationships) ----------
selected_features = ["Price", "Age_08_04", "KM", "HP", "Weight"]
selected_features = [f for f in selected_features if f in df.columns]

sns.pairplot(df[selected_features])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()

# ---------- CATEGORICAL vs NUMERICAL ----------
if "Fuel_Type" in df.columns and "Price" in df.columns:
    plt.figure()
    sns.boxplot(x="Fuel_Type", y="Price", data=df)
    plt.title("Price vs Fuel Type")
    plt.show()


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("ToyotaCorolla - MLR.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Check missing values before handling
print("Missing values before imputation:\n")
print(df.isnull().sum())

# Separate numerical and categorical columns
numerical_features = df.select_dtypes(include=np.number).columns
categorical_features = df.select_dtypes(exclude=np.number).columns

# Impute numerical features with mean (SAFE METHOD)
df[numerical_features] = df[numerical_features].fillna(
    df[numerical_features].mean()
)

# Impute categorical features with mode (SAFE METHOD)
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check missing values after handling
print("\nMissing values after imputation:\n")
print(df.isnull().sum())


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("ToyotaCorolla - MLR.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Handle missing values (safe method)
numerical_features = df.select_dtypes(include=np.number).columns
categorical_features = df.select_dtypes(exclude=np.number).columns

df[numerical_features] = df[numerical_features].fillna(
    df[numerical_features].mean()
)

for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

# --------- ENCODE CATEGORICAL VARIABLES ---------

# One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Display encoded columns
print("Encoded feature columns:\n")
print(df_encoded.columns.tolist())


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load Titanic train dataset
train_df = pd.read_csv("Titanic_train.csv")

# Clean column names
train_df.columns = train_df.columns.str.strip()

# Drop unnecessary columns
train_df = train_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

# Handle missing values
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])

# Encode categorical variables
train_df = pd.get_dummies(train_df, columns=["Sex", "Embarked"], drop_first=True)

# Separate features and target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Feature Scaling (IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Logistic Regression model (increased max_iter)
model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Titanic train dataset
df = pd.read_csv("Titanic_train.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Separate features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------- MODEL TRAINING ---------

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully on training data.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# Load Titanic train dataset
df = pd.read_csv("Titanic_train.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Separate features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# --------- PERFORMANCE METRICS ---------

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_pred_prob))

# --------- ROC CURVE ---------

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Titanic train dataset
df = pd.read_csv("Titanic_train.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Separate features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_scaled, y)

# --------- COEFFICIENT INTERPRETATION ---------

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print(coef_df)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Load the dataset
train_df = pd.read_csv('Titanic_train.csv')

# 2. Preprocessing
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df_encoded.drop('Survived', axis=1)
y = df_encoded['Survived']

# 3. Build Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 4. Extract Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# 5. Visualize (Fixed Seaborn Plot)
plt.figure(figsize=(10, 6))

# UPDATED LINE: Added hue='Feature' and legend=False to fix the warning
sns.barplot(x='Coefficient', y='Feature', data=importance, hue='Feature', palette='RdYlGn', legend=False)

plt.title('Feature Significance (Logistic Regression Coefficients)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# 6. Display Table with Math Formatting
from IPython.display import display, Markdown

table_md = """
| Feature | Coefficient | Interpretation |
| :--- | :---: | :--- |
"""
for _, row in importance.iterrows():
    table_md += f"| **{row['Feature']}** | ${row['Coefficient']:.4f}$ | {'Increases survival odds' if row['Coefficient'] > 0 else 'Decreases survival odds'} |\n"

display(Markdown(table_md))

import pickle

# Assuming 'model' is your trained LogisticRegression object
# and 'X.columns' are the features used during training
model_data = {
    "model": model,
    "features": list(X.columns)
}

with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model saved as titanic_model.pkl")