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

# + _kg_hide-output=true _kg_hide-input=true id="Rf3KKixN6lSM"
# !pip install --quiet pycaret==3.2
# !pip install --quiet shapely>=2.0.1

# + id="BQvEZZOs8gnX"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="FuJCvKVJoaTL" outputId="bca30cef-d7ae-4f44-e82e-4a2033835574"
df_train.head()

# + id="BjqO49NS6lRv"
test_id = df_test["index"]

# + [markdown] id="x15sblBN6lRy"
# # <p style="font-family:newtimeroman;font-size:80%;text-align:center;color:#F52549;">Head Of Datasets</p>

# + [markdown] id="DwoMIzkOo_mN"
#

# + id="FqXxnD2h6lRz" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="dab23467-d7ca-4b24-cdc7-4ef15655de63"
df_submission.head()

# + colab={"base_uri": "https://localhost:8080/"} id="LQqVH4Qrb4zR" outputId="aee9e745-c95f-46db-eb4f-9cbcc654b266"
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

print("Test for Github Action part 2")

# + colab={"base_uri": "https://localhost:8080/"} id="mKNvSa7efpj9" outputId="ba5fae69-3028-4c5c-8850-67696d3ff32f"
# Null Values in Train
train_null = df_train.isnull().sum().sum()

#Null Count in Test
test_null = df_test.isnull().sum().sum()

#null Count in Submission
submission_null = df_submission.isnull().sum().sum()

print(f'Null Count in Train: {train_null}')
print(f'Null Count in Test: {test_null}')
print(f'Null Count in Submission: {submission_null}')

# + colab={"base_uri": "https://localhost:8080/"} id="Jj-5hOhrfxMR" outputId="637f5aa1-d499-4ea6-a6ce-bc95688b6fb7"
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

# + colab={"base_uri": "https://localhost:8080/"} id="8BUVAnVGf4G5" outputId="e7a1874a-886b-4e04-d21f-8f808bcaea98"
df_train.info()

# + colab={"base_uri": "https://localhost:8080/", "height": 300} id="RYQ9PD8gf5j2" outputId="f61e0223-1132-4bf8-9abd-97dafd49b1d1"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="huXtxWkmLKEv" outputId="87059ac5-4d3c-4162-df5e-7faba494defa"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'Age', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="kz8YRzX0LzNb" outputId="c0284bc5-1dcb-4c05-9640-afe3b6b8c601"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'Pregnancies', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="ZhSCKy4PMbCs" outputId="7a611e90-b70d-40a9-9b62-ec7c16009c52"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'Glucose', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="8xeZzVUSMmh5" outputId="07235329-a78a-47f6-f6a0-0580c36ef9fe"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'SkinThickness', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="E4TOU12qM2P3" outputId="7462974f-d71c-4070-fbdf-66bfe5e61fbe"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'BloodPressure', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="mrt_7mQWNM-M" outputId="483544a9-71af-4224-c0b8-85d46ffb9a06"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 25)
fig.map(sns.histplot, 'Insulin', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="pAwEXKH5Nd6j" outputId="355a4007-c48b-431f-d883-1e8228919d30"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'DiabetesPedigreeFunction', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="rMIhaLN0NkE5" outputId="7e557491-694d-493b-cc6c-b9594947620c"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'BMI', bins=40, kde=False)

# + id="xi1dp1EF4zcQ"
#plot_numerical_distribution_with_hue(df_train, num_cols, 'Outcome')

# + colab={"base_uri": "https://localhost:8080/", "height": 974} id="P-yCyvep53sG" outputId="1574c3c9-b764-4314-af3a-d48f9d987c42"
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

# + id="CzUyFBmw6lSG" colab={"base_uri": "https://localhost:8080/"} outputId="73091281-46a5-4775-b05b-d5e83ad020c2"
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

# + id="I_IJk_fW6lSH" colab={"base_uri": "https://localhost:8080/"} outputId="6863f297-90a8-41eb-a0e1-334b0a55b431"
# Shape of Train and Test
print(f"The shape of Train data is {df_train.shape}")
print(f"The shape of Test data is {df_test.shape}")

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="qqhruObgANoA" outputId="37cf87d5-f5ac-49ad-c9ef-48bc8523f1a2"
df_train.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="SZ_-rBvtAT2N" outputId="b8b9d0dd-3e5c-48f6-f11b-0ae36ec745c3"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 418} outputId="1880809b-253b-4893-e0cf-e9b0b344a183" id="Wqx_c9WbrBz9"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
plt.ylim(0, 200)
fig.map(sns.histplot, 'BloodPressure', bins=40, kde=False)

# + id="3rNA6knzvcSN"
df_train = df_train[df_train['BMI'] > 20.0]

# + colab={"base_uri": "https://localhost:8080/", "height": 418} id="K7_lTOVnvlo0" outputId="9e0e3a0e-a2a2-49c5-f230-4f2e329ddd4e"
fig = sns.FacetGrid(df_train, col='Outcome', hue='Outcome', height=4)
fig.map(sns.histplot, 'BMI', bins=40, kde=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="QJzfTvqSm5tR" outputId="6c9cca5a-3345-43ab-feb4-e42f2250067b"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 206} id="lqN7YfqZuKA2" outputId="6eb60496-d606-483a-f65f-705f118dc123"
df_train.head()

# + [markdown] id="lUYg7xrJ6lSM"
# # <p style="font-family:newtimeroman;font-size:100%;text-align:center;color:#F52549;">LightBoost | PyCaret </p>

# + [markdown] id="R-NCPdVG6lSM"
# ![image.png](attachment:3447685d-37cc-40f9-8f77-236f02f8608c.png)

# + id="QYhVk02L6lSN" colab={"base_uri": "https://localhost:8080/", "height": 645} outputId="b0a0b016-4538-4519-cc11-2d8f088de00e"
TARGET = 'Outcome'
# Import
from pycaret.classification import *
# Setup
clf1 = setup(data=df_train, target=TARGET)

# + colab={"base_uri": "https://localhost:8080/", "height": 551, "referenced_widgets": ["5998060cc0cf4e29b0e3d308669bf442", "60363eadeb244569bb6cc998bdbf8a93", "616c0f9868184769a3c2e748b524a16a", "8eee360e00d34cc69e13a20c4b2e44c9", "6a222aacd66a4f68b37a17e314d1eed2", "75cfe8332e244e94b6e09abab6fdaa3b", "b8b0ed6744d0414a94fc1f7798eb9096", "474b572da7bf4e03b7a76c4e5c161a5e", "35473c24fde64cd1a6cd192733135562", "2ebb9768d700484bbc1fe546db68e0cf", "080e77eff3dd466a941daddfe65049bb"]} id="fc2N5DbOlzNW" outputId="11aac894-2842-4e40-94b4-db6c9d0802b2"
 best = compare_models()

# + colab={"base_uri": "https://localhost:8080/"} id="bKSqGfLs37pK" outputId="4a55293b-3a37-41a1-ae8c-fb240f99a9fd"
print(best)

# + colab={"base_uri": "https://localhost:8080/", "height": 914, "referenced_widgets": ["920d2714e35c4e4ab1d884044a55ca2b", "38a51abf1cd64767b443a076bdeafccc", "f2cdf00f4ba54463a52834974ca790a9", "91c530e9a3b0438a9c637baf87254e44", "40ba6d83e2d144a8b6f032e0cc29b186", "951dcc1918384baa887be239907e91e9", "3a8655735efa4e2cadd5406aba890083", "5249c24a13d043ab8ed1b659b508d843", "e07e363e643d4a7e82ad92f21ec88871", "3a5572b7496540dfa22fa02a2090cfdb", "06240e1b58f6400091db639bc6f14814", "04cd6a8c647a4e72afcdfbd9a64b8924", "735c49c9060c44e387486636419a7aa1", "0c9f564e77f7451690cb978a7a0a5f3e", "5b087d7f474749ee9205e84f30a7195d", "41359531f4244645bb50082fc1623aa4", "467a094a69f64d2a82cae839ea7ffaaf", "61fe461cf5934ba5bb358c9cd6f48eb3", "e3ddbb22bcc340ce855227bac7dfea94", "bfb011f8d3a3423190f6376765857675", "5cd28a1115e241b9a8ab27e5fbc7278b", "2e0c3b2ae19949c987c180d081963e8d"]} id="fQADdA9EvcjS" outputId="04de0aff-69f8-4df5-d83a-e3618e50bea5"
model = create_model('lightgbm')
#tuned_model = model
tuned_model = tune_model(model)

# + id="o9qyDCaZxSmc" colab={"base_uri": "https://localhost:8080/", "height": 247, "referenced_widgets": ["e4af12421f2946b99e9b04aa22479ac0", "8ed84e598cf143939b9591c4d05ed0f7", "88726c3a701b486fb94f1364d63516a9", "eb01a3152eb6498fa446600b7522be07", "bf56ad7b8ab44e88ab076a2de8e0a333", "02082e33cd374a0e92c9b8d64ec2a966", "e9e8ce11d3a044d89bca1c0d161d6b2c"]} outputId="2743b5c1-c0fd-4358-d275-fd5b825a9e27"
evaluate_model(tuned_model)

# + id="s1UAz0jm6lSN"
# Finalize Model
f_model = finalize_model(tuned_model)

# + id="AzhwbWG-6lSN" colab={"base_uri": "https://localhost:8080/", "height": 592} outputId="b0cb38f6-dd71-4ea4-eba8-e9a579995bd8"
# Best Model Prediction
predict_model(f_model,
              round = 10)

# + id="tbc-71bC6lSN" colab={"base_uri": "https://localhost:8080/", "height": 247, "referenced_widgets": ["0409c24ff049492ba9d5a8f154af0f31", "553c5411fd704a66aad545d0967b2356", "1b2d9586b07a474db93eaac8d37516e7", "f6f01f18a83142a08fbf5f6c2a949a65", "2a1915970dcd471b8d30f4b01d9f1d97", "3f1a24088e7a401c8f1187832f241f86", "bb01450cdf464ca287c67bd4417a6921"]} outputId="e2c12a3c-c449-4aed-9233-5e7661a88915"
# Evaluate Best Final Model
evaluate_model(f_model)

# + [markdown] id="NxGM5NQU6lSO"
# https://github.com/scipy/scipy/blob/v1.10.1/scipy/_lib/_util.py#L153
# 上記を見る限り、prod は scipy v1.10.1 に存在しているように見える

# + id="PozQGnmTJYwb"
#df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
#inf_presence = df_test.isin([np.inf, -np.inf]).values.any()
#print(inf_presence)

# + id="tr-8IINm6lSO" colab={"base_uri": "https://localhost:8080/", "height": 248} outputId="e18dab52-81f1-44e0-8c4f-f54bf2bd180b"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 143} id="5Q6BRMm5FNnh" outputId="a4859112-a96a-4abd-ceec-76590e4fd9a5"
p_Light.head(3)

# + id="lNbSbYud6lSO" colab={"base_uri": "https://localhost:8080/", "height": 206} outputId="32bb2479-dec4-4a3d-ca2b-be22b52a21cb"
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
