import pandas as pd
import numpy as np 
#import matplotlib.pylot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#sns.set()
#%matplotlib inline
df=pd.read_csv("SalaryData.csv")
print(df.head())
print(df.shape)
df.isnull().values.any()
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
df_copy = train_set.copy()
print(df_copy.shape)
print(df_copy.head())
#print(df_copy.describe())
#print(df_copy.corr())
test_set_full = test_set.copy()
test_set = test_set.drop(["Salary"], axis=1)
test_set.head()
train_labels = train_set["Salary"]
train_labels.head()
train_set_full = train_set.copy()

train_set = train_set.drop(["Salary"], axis=1)
train_set.head()

lin_reg = LinearRegression()
lin_reg.fit(train_set, train_labels)

salary_pred = lin_reg.predict(test_set)

print(salary_pred)

