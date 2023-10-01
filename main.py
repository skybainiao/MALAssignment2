import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


file_path = 'C:/Users/45527/Downloads/exoplanet_dataset.csv'
df = pd.read_csv(file_path)


print("Data Overview:")
print(df.head())
print(df.info())
print(df.describe())


df = df.dropna(axis=1, how='all')


imputer = SimpleImputer(strategy="mean")
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])


le = LabelEncoder()
df['koi_disposition'] = le.fit_transform(df['koi_disposition'])


X = df.select_dtypes(include=[np.number]).drop(['koi_disposition'], axis=1)
y = df['koi_disposition']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
pipeline.fit(X_train, y_train)


scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f'Mean Cross Validation Score: {scores.mean()}')


y_pred = pipeline.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))


feature_importances = clf.feature_importances_
features = list(X.columns)
feature_importance_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances: \n", feature_importance_df)


selected_features = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_period']  # 替换为你的实际特征列名
df[selected_features].hist(bins=30, figsize=(10, 7))
plt.suptitle('Histograms of Selected Features')


correlation_matrix = df[selected_features].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Selected Features')
plt.show()
