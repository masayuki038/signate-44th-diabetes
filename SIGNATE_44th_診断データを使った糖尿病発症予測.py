# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/masayuki038/signate-44th-diabetes/blob/main/SIGNATE_44th_%E8%A8%BA%E6%96%AD%E3%83%87%E3%83%BC%E3%82%BF%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9F%E7%B3%96%E5%B0%BF%E7%97%85%E7%99%BA%E7%97%87%E4%BA%88%E6%B8%AC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + _kg_hide-output=true _kg_hide-input=true id="Rf3KKixN6lSM" colab={"base_uri": "https://localhost:8080/"} outputId="b34a5b37-22b5-459d-8fb4-c8ff249e1091"
# !pip install --quiet pycaret==3.2
# !pip install --quiet shapely>=2.0.1

# + id="BQvEZZOs8gnX" colab={"base_uri": "https://localhost:8080/"} outputId="ddac62bf-15c6-4843-ad26-0997cb40bb9d"
# !pip install --quiet catboost

# + [markdown] id="p5lEGNny6lRs"
# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Importing Libraries</p>

# + id="kXYvEcCd6lRt"
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

# + [markdown] id="95E58TS76lRu"
# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Data Loading and Overview</p>

# + id="pksZMC8P6lRu"
# Load Submission Data
df_submission = pd.read_csv('/var/SIGNATE/44th/sample_submit.csv')
# Load test Data
df_test = pd.read_csv('/var/SIGNATE/44th/test.csv')
# Load Train Dataset and show head of Data
#Train Data
df_train = pd.read_csv('/var/SIGNATE/44th/train.csv')

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="FuJCvKVJoaTL" outputId="5317310c-9bae-44b5-f94d-c54cd388476e"
df_train.head()

# + id="BjqO49NS6lRv"
test_id = df_test["index"]

# + [markdown] id="x15sblBN6lRy"
# # <p style="font-family:newtimeroman;font-size:80%;text-align:center;color:#F52549;">Head Of Datasets</p>

# + [markdown] id="DwoMIzkOo_mN"
#

# + id="FqXxnD2h6lRz" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="9fd4a043-5bc6-4e1e-fc8e-9d1a519e62d7"
df_submission.head()

# + colab={"base_uri": "https://localhost:8080/"} id="LQqVH4Qrb4zR" outputId="8d761831-d7e0-4e07-d3d8-afaa797e55a9"
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

# + colab={"base_uri": "https://localhost:8080/"} id="mKNvSa7efpj9" outputId="fd05baf1-7850-4c74-ad94-573b51d6f0b8"
# Null Values in Train
train_null = df_train.isnull().sum().sum()

#Null Count in Test
test_null = df_test.isnull().sum().sum()

#null Count in Submission
submission_null = df_submission.isnull().sum().sum()

print(f'Null Count in Train: {train_null}')
print(f'Null Count in Test: {test_null}')
print(f'Null Count in Submission: {submission_null}')

# + colab={"base_uri": "https://localhost:8080/"} id="Jj-5hOhrfxMR" outputId="56148bff-1c4a-4cf5-90a8-44d9d0f08f73"
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

# + colab={"base_uri": "https://localhost:8080/"} id="8BUVAnVGf4G5" outputId="41d1cf72-57c9-47ed-f9d8-8491a510e5b8"
df_train.info()

# + colab={"base_uri": "https://localhost:8080/", "height": 300} id="RYQ9PD8gf5j2" outputId="3e02516e-7314-4037-f47f-2686cb1389ad"
df_train.describe()


# + id="YAKpuV9BhMP5"
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


# + id="KNM1ycoihNqc"
num_cols = ['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age']
#plot_numerical_distribution_with_hue(df_train, num_cols, 'Outcome')

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="huXtxWkmLKEv" outputId="aa7651b0-6f67-46e5-876d-2325c29132f6"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'Age', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="kz8YRzX0LzNb" outputId="f0255a75-7766-4086-e390-56d963b05b31"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'Pregnancies', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="ZhSCKy4PMbCs" outputId="6b631463-dbf1-4e22-e019-7febd0c8c674"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'Glucose', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="8xeZzVUSMmh5" outputId="213f5bc5-6240-4f5e-fdd9-b66081d6ba10"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'SkinThickness', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="E4TOU12qM2P3" outputId="b0a2b0a0-e4a3-4657-e1a0-c921835c37a6"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'BloodPressure', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="mrt_7mQWNM-M" outputId="a41ae5e3-6ac4-43b6-aab9-098496c1fd5e"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 25)
fig.map(sns.histplot, 'Insulin', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="pAwEXKH5Nd6j" outputId="f1f97bbd-e8bf-46bc-e629-6c74440fcac6"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'DiabetesPedigreeFunction', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="rMIhaLN0NkE5" outputId="2021fef7-bd1c-4902-d77c-0ba7dc77cd69"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'BMI', bins=40, kde=False)

# + id="xi1dp1EF4zcQ"
#plot_numerical_distribution_with_hue(df_train, num_cols, 'Outcome')

# + colab={"base_uri": "https://localhost:8080/", "height": 974} id="P-yCyvep53sG" outputId="3b1e6d69-3dee-4eb9-afd9-e44e3e043797"
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

# + [markdown] id="aGG80YRB6lSG"
# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Correlation Analysis</p>

# + id="CzUyFBmw6lSG" colab={"base_uri": "https://localhost:8080/"} outputId="739509c4-8056-4e16-ccac-9120e9c71cc3"
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

# + [markdown] id="ntJAD-Eb6lSG"
# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">Model Building</p>

# + id="mj0P0m4B6lSH"
# Dropping Null Values to | Skips any Error In Next Steps
df_train.dropna(inplace=True)

# + id="NY1eUL1e6lSH"
# Dropping Some Columns From df_train
outcome = df_train['Outcome']
df_train.drop(['index'], axis=1, inplace=True)
df_test.drop(['index'], axis=1, inplace=True)

# + id="lepKCIeLj-0w"
#train_x, test_x, train_y, test_y = train_test_split(
#    df_train.drop('Outcome', axis=1), outcome, test_size=0.3, random_state=42)

# + id="I_IJk_fW6lSH" colab={"base_uri": "https://localhost:8080/"} outputId="68853b6a-811e-4592-969d-940713514b39"
# Shape of Train and Test
print(f"The shape of Train data is {df_train.shape}")
print(f"The shape of Test data is {df_test.shape}")

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="qqhruObgANoA" outputId="c0738d2c-5847-444b-e766-8655a217b051"
df_train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="SZ_-rBvtAT2N" outputId="5f4638c1-e931-422c-baa9-ff110cf43d13"
df_test.head()

# + id="43E9dgRuiniB"
#train_x.head()

# + id="saDfX4vliq5a"
#train_y.head()

# + id="ICZ-yLbbn7Y0"
# clean up

# 本来、Insulin が高ければ糖尿病の傾向は高まるはずだが、ヒストグラムを見る限りその傾向は見られないので削除
#df_train.drop(['Insulin'], axis=1, inplace=True)
#df_test.drop(['Insulin'], axis=1, inplace=True)

# SkinThickness も相関が見られないので削除
df_train.drop(['SkinThickness'], axis=1, inplace=True)
df_test.drop(['SkinThickness'], axis=1, inplace=True)

# + id="mJZGjDs3tF7V"
#df_train[df_train['SkinThickness'] != 0]
df_train = df_train[df_train['BloodPressure'] != 0]

# + colab={"base_uri": "https://localhost:8080/", "height": 424} outputId="b556f6a4-62fc-474e-86ab-d3cd2c700d90" id="Wqx_c9WbrBz9"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'BloodPressure', bins=40, kde=False)

# + id="3rNA6knzvcSN"
df_train = df_train[df_train['BMI'] > 20.0]

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="K7_lTOVnvlo0" outputId="dfff25ad-5200-4ace-bc6e-c400040257af"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'BMI', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="QJzfTvqSm5tR" outputId="9d537aa7-39bc-4ea2-8f38-59be0561d49a"
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

# + id="58uDRGfRpl40"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="lqN7YfqZuKA2" outputId="8364de5d-9eb2-44d8-cd42-1da210178e61"
df_train.head()

# + [markdown] id="lUYg7xrJ6lSM"
# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">LightBoost | PyCaret </p>

# + [markdown] id="R-NCPdVG6lSM"
# ![image.png](attachment:3447685d-37cc-40f9-8f77-236f02f8608c.png)

# + id="QYhVk02L6lSN" colab={"base_uri": "https://localhost:8080/", "height": 645} outputId="4acf4cab-6517-4e13-cd59-12e3de222dfb"
TARGET = 'Outcome'
# Import
from pycaret.classification import *
# Setup
clf1 = setup(data=df_train, target=TARGET)

# + colab={"base_uri": "https://localhost:8080/", "height": 551, "referenced_widgets": ["e1f146bd798d4f579a4a056d5a0d1c06", "eb67dbdd04cd4932a773c2c6836beebb", "a22b5e2619e54a4e803a8eedad29998c", "6ead5878e66c4b4baf89916a57ca5870", "bdf8d860994b41d283a2892c80378889", "4c4170847064445dab9f70b81c960236", "4ad23f5960274538892821678921558c", "2724407451b34bd0bc20e4234f4b1589", "9741b556066042f0959b8c9ebf773c21", "64b0b62c86e84fd2886861f0c4635832", "ad6fc5f7ee48425e9cf271fbb2800ed0"]} id="fc2N5DbOlzNW" outputId="6b0a7f59-1637-475b-d373-b6977e8d0f5c"
 best = compare_models()

# + colab={"base_uri": "https://localhost:8080/"} id="bKSqGfLs37pK" outputId="aa8a61e2-2831-4737-f040-6b1162094a60"
print(best)

# + colab={"base_uri": "https://localhost:8080/", "height": 914, "referenced_widgets": ["9d986e5548904bf2ac5af3a097aa4fdd", "90c529bf7c874791bc8d2f5b86b96756", "41bba87a3f0c443c8ee7bf2267b4694b", "6633b5678802455b8646a424e2e6121a", "4e998dfbf9be491495952e620455a400", "5c3a43a53f7246acbbff6b581c40c71b", "d7814e25ccd54476bf41098a5f593b66", "5b9f398614e548cab9faf548eb4a1fdd", "4a04a0d3c0024b7b85206f10c972a2a1", "65331fb923874327aa0df32b5ea2f76d", "8131ec1ad321491c96db7e881db3a35c", "ef77016cd2c446c690a7ad0b512e966f", "cd8df788ce704a2faa013db6ee86d13b", "72063aef1bf14b4f9d6704881888c98a", "18284908263641a1b7696bc579c06435", "a5ca1c9b544c4cf7971ae93dd6d34f90", "338fa266ba884d00a8578792855b31ae", "b498f3099251400686f3c28b93a56a15", "3142959ed8a34b37a69167d1997f54d0", "d749cc3e843f4931a10810436f567433", "f3bf7f6dd10c4308a083614cd3b6a048", "f0ec399fcbf94755b7ecc3969a77de25"]} id="fQADdA9EvcjS" outputId="f79cde49-1354-4ba7-ee45-98554e6115fd"
model = create_model('lightgbm')
#tuned_model = model
tuned_model = tune_model(model)

# + id="o9qyDCaZxSmc" colab={"base_uri": "https://localhost:8080/", "height": 247, "referenced_widgets": ["490f0f4d1be7435faa1f67252b1daef4", "0f129ea28811415cb2294ea9df4ba5a5", "0bdefa769b9247cbb9d1d35fe7e1154d", "a9cf7190e194448d971d59417b13d5e6", "0e7a9b0b2db64733a0e1229120a5a4de", "5287d6960f7247beb296afba095177e2", "129bb8ce8317481db51267255775d1e4"]} outputId="6186fd9b-7f15-4a67-e5ac-ba73d9947031"
evaluate_model(tuned_model)

# + id="s1UAz0jm6lSN"
# Finalize Model
f_model = finalize_model(tuned_model)

# + id="AzhwbWG-6lSN" colab={"base_uri": "https://localhost:8080/", "height": 592} outputId="55852ef4-fee9-41df-db5b-a4e270522785"
# Best Model Prediction
predict_model(f_model,
              round = 10)

# + id="tbc-71bC6lSN" colab={"base_uri": "https://localhost:8080/", "height": 247, "referenced_widgets": ["551804e54ac747f3853b0a9429e404d1", "ae0932090ed5499888a25c2b0a9de82d", "b68a9f5626324a65875d740f6c6cc35c", "87c7420d7e5e4c62974f3ca46094352e", "69df239a2c1246abb5da3a0ab3a8a97c", "505c39b82ccb436f99a5ffee7cfc7f0f", "57d9ae4e63e14159aee7290ebd0974a1"]} outputId="d8217e97-b057-49d5-97ac-3e5c68ba2764"
# Evaluate Best Final Model
evaluate_model(f_model)

# + [markdown] id="NxGM5NQU6lSO"
# https://github.com/scipy/scipy/blob/v1.10.1/scipy/_lib/_util.py#L153
# 上記を見る限り、prod は scipy v1.10.1 に存在しているように見える

# + id="PozQGnmTJYwb"
#df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
#inf_presence = df_test.isin([np.inf, -np.inf]).values.any()
#print(inf_presence)

# + id="tr-8IINm6lSO" colab={"base_uri": "https://localhost:8080/", "height": 248} outputId="af73a7ec-cde6-402f-df0b-f157ae27d6c5"
# Making Prediction DataFrame
p_Light = predict_model(f_model,
                        data=df_test,
                        raw_score=True,
                        round = 10
                       )
# Head Prediction _df
p_Light.head(3)

# + id="zWIvtZgSCFRn"
p_Light['Outcome'] =p_Light.apply(
    lambda row: 0 if (row['prediction_score_0'] > row['prediction_score_1']) else 1,
    axis=1
)

# + colab={"base_uri": "https://localhost:8080/", "height": 143} id="5Q6BRMm5FNnh" outputId="c767f5bb-a5aa-4d53-86c9-87daf0e3e756"
p_Light.head(3)

# + id="lNbSbYud6lSO" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="6a03898a-c816-4a0e-bd5a-f5dedabe718d"
# Making Submission DF
submission_df = pd.DataFrame({
    'index': test_id,
    'Outcome': p_Light['Outcome']
})
# Head Submission
submission_df.head(5)

# + id="P2ABf0Lo6lSO"
# Submission Save
submission_df.to_csv("/var/SIGNATE/44th/submission.csv", header=False, index=False)
