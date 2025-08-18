#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation & Insights System
# **Author:** PalakPandey | **Batch:** BatchX
# **Created:** 2025-08-18
# 
# **Objective:**
# - Analyze movie data (genres, ratings, runtime, votes, budget, revenue)
# - Predict whether a movie is likely to be a **Hit** or **Flop** using ML
# - Provide visual and statistical insights
# - Demonstrate skills in data cleaning, EDA, visualization, regression, and preprocessing
# 
# **Dataset:** `movies_synthetic_dataset.csv` (synthetic for demo)
# You can replace it with Kaggle **IMDB 5000 Movie Dataset** with minor column mapping.
# 

# In[6]:


# If running in a fresh environment, uncomment to install packages
# !pip install pandas numpy scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
from IPython.display import display



# ## 1. Data Loading & Cleaning

# In[4]:


# Load dataset
import os
csv_path = csv_path = "c:\\Users\\dpand\\Dropbox\\PC\\Downloads\\archive (1)\\tmdb_5000_movies.csv"
csv_path = csv_path = "C:\\Users\\dpand\\Dropbox\\PC\\Downloads\\archive (1)\\tmdb_5000_credits.csv"
csv_path = csv_path = "C:\\Users\\dpand\\OneDrive\\Desktop\\Movie recomendation system\\archive (1)\\movies_synthetic_dataset.csv"

df = pd.read_csv(csv_path)

print("Shape:", df.shape)
df.head()


# In[ ]:


# Investigate missing values
df.isna().sum()


# In[ ]:


# Basic cleaning
# Strategy:
# - For numeric columns: fill NaNs with median
# - For categorical columns: fill NaNs with mode
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isna().sum()


# ## 2. Feature Engineering

# In[ ]:


# Encode target: Hit -> 1, Flop -> 0
df['HitFlag'] = (df['Hit'].str.lower() == 'yes').astype(int)

# Label encode Genre for quick models; also build One-Hot later in pipelines
le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df['Genre'])

# Normalize numeric features (we'll do this in pipeline for models, shown here just for illustration)
from sklearn.preprocessing import MinMaxScaler
scaler_demo = MinMaxScaler()
df_scaled_demo = df.copy()
df_scaled_demo[['Runtime','Budget_M','Votes','Rating','Revenue_M']] = scaler_demo.fit_transform(df_scaled_demo[['Runtime','Budget_M','Votes','Rating','Revenue_M']])
df_scaled_demo.head()


# ## 3. Exploratory Data Analysis (EDA)

# In[ ]:


# Genre distribution
genre_counts = df['Genre'].value_counts().sort_values(ascending=False)
genre_counts


# In[ ]:


# Plot 1: Bar chart - Top Genres
plt.figure()
genre_counts.plot(kind='bar')
plt.title('Top Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()


# In[ ]:


# Rating trends (histogram)
plt.figure()
plt.hist(df['Rating'], bins=20)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Budget vs Revenue scatter with hit coloring
plt.figure()
colors = df['HitFlag'].map({1:'tab:green', 0:'tab:red'})
plt.scatter(df['Budget_M'], df['Revenue_M'], c=colors, alpha=0.6)
plt.title('Budget vs Revenue (colored by Hit)')
plt.xlabel('Budget (Millions)')
plt.ylabel('Revenue (Millions)')
plt.show()


# In[ ]:


# Boxplot: Budget vs Hit
plt.figure()
df.boxplot(column='Budget_M', by='Hit')
plt.title('Budget by Hit/Flop')
plt.suptitle('')
plt.xlabel('Hit')
plt.ylabel('Budget (Millions)')
plt.show()


# In[ ]:


# Heatmap: Correlation matrix
plt.figure()
corr = df[['Runtime','Budget_M','Votes','Rating','Revenue_M','HitFlag','Genre_encoded']].corr()
import seaborn as sns
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ## 4. Statistical Summary

# In[ ]:


summary_stats = df[['Runtime','Budget_M','Votes','Rating','Revenue_M']].agg(['mean','median','std']).T
corr_matrix = df[['Runtime','Budget_M','Votes','Rating','Revenue_M','HitFlag','Genre_encoded']].corr()
print("Summary Statistics:")
display(summary_stats)
print("\nCorrelation Matrix:")
display(corr_matrix)


# ## 5. Model 1: Logistic Regression – Predict Hit/Flop

# In[ ]:


# Features and target
X = df[['Genre','Runtime','Budget_M','Votes','Rating','Revenue_M']]
y = df['HitFlag']

# Define preprocessing for columns
numeric_features = ['Runtime','Budget_M','Votes','Rating','Revenue_M']
categorical_features = ['Genre']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
log_reg = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:,1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ## 6. Model 2: Linear Regression – Predict Rating

# In[5]:


# Predict Rating from other features
X2 = df[['Genre','Runtime','Budget_M','Votes','Revenue_M']]
y2 = df['Rating']

numeric_features2 = ['Runtime','Budget_M','Votes','Revenue_M']
categorical_features2 = ['Genre']

preprocess2 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features2),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features2)
    ])

lin_reg = Pipeline(steps=[
    ('preprocess', preprocess2),
    ('model', LinearRegression())
])

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

lin_reg.fit(X2_train, y2_train)
y2_pred = lin_reg.predict(X2_test)

mae = mean_absolute_error(y2_test, y2_pred)
rmse = np.sqrt(mean_squared_error(y2_test, y2_pred))
r2 = r2_score(y2_test, y2_pred)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2: {r2:.3f}")

# Plot Prediction vs Actual
plt.figure()
plt.scatter(y2_test, y2_pred, alpha=0.6)
plt.title('Linear Regression: Actual vs Predicted Rating')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], 'k--')
plt.show()


# ## 7. Additional Visualizations

# In[ ]:


# Histogram of Runtime
plt.figure()
plt.hist(df['Runtime'], bins=20)
plt.title('Runtime Distribution')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Frequency')
plt.show()


# ## 8. Code Quality & Structure – Utility Functions

# In[ ]:


def summarize_dataframe(dataframe):
    return {
        "rows": dataframe.shape[0],
        "cols": dataframe.shape[1],
        "columns": list(dataframe.columns),
        "missing_values": dataframe.isna().sum().to_dict()
    }

summary_info = summarize_dataframe(df)
summary_info


# ## 9. Summary & Insights

# **Key Takeaways:**
# - Genres with the highest counts in this dataset appear in the bar chart.
# - Ratings tend to cluster around the 6–8 range; higher ratings correlate positively with Hit.
# - Budget and Revenue are positively correlated; Hit movies typically have revenues exceeding budgets.
# - Logistic Regression provides a baseline classifier for Hit/Flop prediction (see metrics).
# - Linear Regression offers a baseline for predicting **Rating** from other features.
# - You can redefine **Hit** (e.g., `Revenue_M > Budget_M` or `Rating > 7.5`) and re-run the notebook to compare.
# 
# **Next Steps (Optional):**
# - Try alternative models (e.g., RandomForest, XGBoost) and compare.
# - Add cross-validation and hyperparameter tuning.
# - Swap in the Kaggle IMDB dataset and map its columns to this schema.
# - Create a PDF slide/report with the best insights and model results.
# 

# In[ ]:





# In[ ]:





# In[ ]:




