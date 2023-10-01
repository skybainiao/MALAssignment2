import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df = pd.read_csv('C:/Users/45527/Downloads/exoplanet_dataset.csv')


print(df.head())
print(df.info())
print(df.describe())
df.hist(bins=50, figsize=(20,15))
plt.show()


print(df.isnull().sum())
df.dropna(inplace=True)


selected_features = [
    'koi_fpflag_nt',
    'koi_fpflag_ss',
    'koi_fpflag_co',
    'koi_fpflag_ec',
    'koi_period',
    'koi_time0bk',
    'koi_impact',
    'koi_duration',
    'koi_depth',
    'koi_prad',
    'koi_teq',
    'koi_insol',
    'koi_model_snr',
    'koi_steff',
    'koi_slogg',
    'koi_srad'
]
X = df[selected_features]
le = LabelEncoder()
y = le.fit_transform(df['koi_disposition'])







# 检查数据集中是否有足够的样本以进行划分
if len(df) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    # 检查训练集和测试集的样本数量
    print("训练集样本数:", len(X_train))
    print("测试集样本数:", len(X_test))
else:
    print("数据集中没有足够的样本来进行划分，请检查数据集或数据加载步骤。")
