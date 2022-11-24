# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:12:27 2022

@author: faikm
"""

import pandas as pd
import numpy as np

train_data = pd.read_csv("train.csv")

toplam = train_data.isnull().sum().sort_values(ascending=False)
yuzde = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)

eksik_veriler = pd.concat([toplam, yuzde], axis=1, keys=(["Toplam", "Yuzde"]))
train_data = train_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','Id'], axis=1)

import matplotlib.pyplot as plt
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16.0
fig_size[1] = 4.0

x = train_data['SalePrice']
plt.hist(x, bins=400)
plt.ylabel('salecPrice')
plt.show()

def aykırı_degerler():
    filtre = [h for h in (train_data['SalePrice']) if(h <450000)]
    return filtre
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 16.0
fig_size[0] = 10

filtre = aykırı_degerler()
plt.hist(filtre, 50)
fig_size[0] = 16.0
fig_size[1] = 8.0
plt.show()

df_aykırsız = pd.DataFrame(filtre)

train_data = train_data[train_data.SalePrice < 450000]

X_train = train_data.drop(['SalePrice'], axis=1)
y_labels = train_data['SalePrice']


X_train = X_train.apply(lambda x: x.fillna(x.value_counts().index[0])) # En sık sayısal ile doldur.

kategoriler = ["GarageFinish", "BsmtQual", "GarageType", "GarageQual", "GarageCond", "BsmtCond", "BsmtExposure", "BsmtFinType1", "FireplaceQu" ]

for i in kategoriler:
    X_train = X_train.fillna(X_train[i].value_counts().index[0])
    
X_train = pd.get_dummies(X_train, columns=[])    

#Kategori verileri sahte sütunlar oluşturma
X_train = pd.get_dummies(X_train, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])     

X_train = X_train.drop(['Condition2_RRAe','Exterior2nd_Other','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Heating_Floor','Heating_OthW','Electrical_Mix','GarageQual_Ex', 'Exterior1st_Stone','Utilities_NoSeWa'], axis=1)

#Test Verilerle işlem yapma

test_data = pd.read_csv("test.csv")

test_toplam = test_data.isnull().sum().sort_values(ascending=False)
test_yuzde = (test_data.isnull().sum() / test_data.isnull().count()).sort_values(ascending=False)

eksik_veriler_test = pd.concat([test_toplam, test_yuzde], axis=1, keys=["Test_Toplam", "Test_yuzde"])

test_data = test_data.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)

test_data = test_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

kategoriler_test = ["GarageFinish", "BsmtQual", "FireplaceQu", "GarageType", "GarageQual", "GarageCond", "GarageFinish", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtUnfSF"]

for i in kategoriler_test:
    test_data = test_data.fillna(test_data[i].value_counts().index[0])
    
test_data = pd.get_dummies(test_data, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'])     

X_test = test_data.drop(['Id'], axis=1)

from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score



xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb.fit(X_train, y_labels)
xgb_cv = cross_val_score(xgb,X_train, y_labels, cv=10, )
print(np.median(xgb_cv))
xgb_pred = xgb.predict(X_test)


submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': xgb_pred})

submission.to_csv("Kaggle_HousePrices.csv",index=False)
