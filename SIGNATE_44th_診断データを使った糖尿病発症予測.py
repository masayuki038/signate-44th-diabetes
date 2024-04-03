# <a href="https://colab.research.google.com/github/masayuki038/signate-44th-diabetes/blob/main/SIGNATE_44th_%E8%A8%BA%E6%96%AD%E3%83%87%E3%83%BC%E3%82%BF%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E7%B3%96%E5%B0%BF%E7%97%85%E7%99%BA%E7%97%87%E4%BA%88%E6%B8%AC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# !pip install --quiet pycaret==3.2
# !pip install --quiet shapely>=2.0.1

# !pip install --quiet catboost

# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Importing Libraries</p>

# +
# import libraries

# 1. to handle the data
import pandas as pd
import numpy as np

# to visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# this is for jupyter notebook to show the plot in the notebook itself instead of opening a new window
# %matplotlib inline

# To preprocess the data
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler , QuantileTransformer

# machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.multioutput import MultiOutputClassifier
#Model
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric

#Evaluation
from sklearn.metrics import roc_auc_score

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set display options for maximum columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# -

# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Data Loading and Overview</p>

# Load Submission Data
df_submission = pd.read_csv('/var/SIGNATE/44th/sample_submit.csv')
# Load test Data
df_test = pd.read_csv('/var/SIGNATE/44th/test.csv')
# Load Train Dataset and show head of Data
#Train Data
df_train = pd.read_csv('/var/SIGNATE/44th/train.csv')

df_train.head()

test_id = df_test["index"]

# # <p style="font-family:newtimeroman;font-size:80%;text-align:center;color:#F52549;">Head Of Datasets</p>

#

df_submission.head()

# +
# ここから再開
num_train_rows, num_train_columns = df_train.shape

num_test_rows, num_test_columns = df_test.shape

num_submission_rows, num_submission_columns = df_submission.shape

print("Training Data:")
print(f"Number of Rows: {num_train_rows}")
print(f"Number of Columns: {num_train_columns}\n")

print("Test Data:")
print(f"Number of Rows: {num_test_rows}")
print(f"Number of Columns: {num_test_columns}\n")

print("Submission Data:")
print(f"Number of Rows: {num_submission_rows}")
print(f"Number of Columns: {num_submission_columns}")

# +
# Null Values in Train
train_null = df_train.isnull().sum().sum()

#Null Count in Test
test_null = df_test.isnull().sum().sum()

#null Count in Submission
submission_null = df_submission.isnull().sum().sum()

print(f'Null Count in Train: {train_null}')
print(f'Null Count in Test: {test_null}')
print(f'Null Count in Submission: {submission_null}')

# +
# Count duplicate rows in train_data
train_duplicates = df_train.duplicated().sum()

# Count duplicate rows in test_data
test_duplicates = df_test.duplicated().sum()

# Count duplicate rows in original_data
submission_duplicates = df_submission.duplicated().sum()

# Print the results
print(f"Number of duplicate rows in train_data: {train_duplicates}")
print(f"Number of duplicate rows in test_data: {test_duplicates}")
print(f"Number of duplicate rows in test_data: {submission_duplicates}")
# -

df_train.info()

df_train.describe()


# Function to Plot Numerical Distribution With Respect to Target
def plot_numerical_distribution_with_hue(data, num_cols, hue_col, figsize=(60, 30), dpi=300):
    # Create subplots
    fig, ax = plt.subplots(2, 4, figsize=figsize, dpi=dpi)
    ax = ax.flatten()

    # Loop through each column and plot the distribution with hue
    for i, column in enumerate(num_cols):
        print(f'{i}, {column}')
        sns.histplot(data=data, x=column, hue=hue_col, ax=ax[i], kde=True, palette='Set2')
        ax[i].set_title(f'{column} Distribution', size=14)
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(None)

    # Set Tight Layout
    plt.tight_layout()

    # Show the plot
    plt.show()


num_cols = ['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age']
#plot_numerical_distribution_with_hue(df_train, num_cols, 'Outcome')

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'Age', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'Pregnancies', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'Glucose', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'SkinThickness', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'BloodPressure', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 25)
fig.map(sns.histplot, 'Insulin', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'DiabetesPedigreeFunction', bins=40, kde=False)

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'BMI', bins=40, kde=False)

# +
#plot_numerical_distribution_with_hue(df_train, num_cols, 'Outcome')

# +
# Select only numeric columns
numeric_df_train = df_train.select_dtypes(include='number')

# Compute the correlation matrix
correlation_matrix = numeric_df_train.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(25, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Plot', fontsize=22)
plt.tight_layout()
plt.show()
# -

# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Correlation Analysis</p>

# +
# Define the correlation threshold
threshold = 0.2

# Extract upper triangular portion of the correlation matrix
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find highly correlated features
highly_correlated = (upper_tri.abs() > threshold).any()

# Get the names of highly correlated features
correlated_features = upper_tri.columns[highly_correlated].tolist()

# Print the names of highly correlated features
print("Highly Correlated Features:")
print(correlated_features)
# -

# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Model Building</p>

# Dropping Null Values to | Skips any Error In Next Steps
df_train.dropna(inplace=True)

# Dropping Some Columns From df_train
outcome = df_train['Outcome']
df_train.drop(['index'], axis=1, inplace=True)
df_test.drop(['index'], axis=1, inplace=True)

# +
#train_x, test_x, train_y, test_y = train_test_split(
#    df_train.drop('Outcome', axis=1), outcome, test_size=0.3, random_state=42)
# -

# Shape of Train and Test
print(f"The shape of Train data is {df_train.shape}")
print(f"The shape of Test data is {df_test.shape}")

df_train.head()

df_test.head()

# +
#train_x.head()

# +
#train_y.head()

# +
# clean up

# 本来、Insulin が高ければ糖尿病の傾向は高まるはずだが、ヒストグラムを見る限りその傾向は見られないので削除
#df_train.drop(['Insulin'], axis=1, inplace=True)
#df_test.drop(['Insulin'], axis=1, inplace=True)

# SkinThickness も相関が見られないので削除
df_train.drop(['SkinThickness'], axis=1, inplace=True)
df_test.drop(['SkinThickness'], axis=1, inplace=True)
# -

#df_train[df_train['SkinThickness'] != 0]
df_train = df_train[df_train['BloodPressure'] != 0]

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'BloodPressure', bins=40, kde=False)

df_train = df_train[df_train['BMI'] > 20.0]

fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'BMI', bins=40, kde=False)

# +
# feature engineering

# ビンの範囲をリストで指定
age_bins = [0, 19, 39, 59, 79, 89]

# pd.cut を使用して離散的なカテゴリに変換
#df_train['Age_C'] = pd.cut(df_train['Age'], bins=age_bins, labels=['<20', '20<40', '40<60', '60<80', '80<'])
#df_test['Age_C'] = pd.cut(df_test['Age'], bins=age_bins, labels=['<20', '20<40', '40<60', '60<80', '80<'])

# ビンの範囲をリストで指定
bmi_bins = [10, 19.99, 29.99, 39.99, 100]

# pd.cut を使用して離散的なカテゴリに変換
#df_train['BMI_C'] = pd.cut(df_train['BMI'], bins=bmi_bins, labels=['10.00-19.99', '20.00-29.99', '30.00-39.99', '40.00-100.00'])
#df_test['BMI_C'] = pd.cut(df_test['BMI'], bins=bmi_bins, labels=['10.00-19.99', '20.00-29.99', '30.00-39.99', '40.00-100.00'])

# "age"列と"bmi"列の値を足して新しい列"age+bmi"を作成
#df_train['age+preg'] = df_train['Age'] + ((df_train['Pregnancies'] * 4) + 10)
#df_train.drop(['Age', 'BMI'], axis=1, inplace=True)

#df_test['age+preg'] = df_test['Age'] + ((df_test['Pregnancies'] * 4) + 10)
#df_test.drop(['Age', 'BMI'], axis=1, inplace=True)

df_train.head()

# +
# normalization

# Normalizations Using Quantile Transformer
# Intlizer QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')

#cols_for_normal = ['Insulin', 'SkinThickness', 'DiabetesPedigreeFunction']
cols_for_normal = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI']
# Fit and Transform
for col in cols_for_normal:
    df_train[col] = qt.fit_transform(df_train[[col]])
for col in cols_for_normal:
    df_test[col] = qt.fit_transform(df_test[[col]])
# -

df_train.head()

# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">LightBoost | PyCaret </p>

# ![image.png](attachment:3447685d-37cc-40f9-8f77-236f02f8608c.png)

TARGET = 'Outcome'
# Import
from pycaret.classification import *
# Setup
clf1 = setup(data=df_train, target=TARGET)

 best = compare_models()

print(best)

model = create_model('lightgbm')
#tuned_model = model
tuned_model = tune_model(model)

evaluate_model(tuned_model)

# Finalize Model
f_model = finalize_model(tuned_model)

# Best Model Prediction
predict_model(f_model,
              round = 10)

# Evaluate Best Final Model
evaluate_model(f_model)

# https://github.com/scipy/scipy/blob/v1.10.1/scipy/_lib/_util.py#L153
# 上記を見る限り、prod は scipy v1.10.1 に存在しているように見える

# +
#df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
#inf_presence = df_test.isin([np.inf, -np.inf]).values.any()
#print(inf_presence)
# -

# Making Prediction DataFrame
p_Light = predict_model(f_model,
                        data=df_test,
                        raw_score=True,
                        round = 10
                       )
# Head Prediction _df
p_Light.head(3)

p_Light['Outcome'] =p_Light.apply(
    lambda row: 0 if (row['prediction_score_0'] > row['prediction_score_1']) else 1,
    axis=1
)

p_Light.head(3)

# Making Submission DF
submission_df = pd.DataFrame({
    'index': test_id,
    'Outcome': p_Light['Outcome']
})
# Head Submission
submission_df.head(5)

# Submission Save
submission_df.to_csv("/var/SIGNATE/44th/submission.csv", header=False, index=False)
