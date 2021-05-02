

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Train.csv")

dataset

sns.countplot(x="price",hue ="coupon_code",data = dataset)

sns.countplot(x="transaction_type",hue ="payment_code",data = dataset)

dataset.isnull()

sns.heatmap(dataset.isnull(),yticklabels=False,cmap='viridis')

cate_customer_id = dataset["customer_id"].copy()

cate_customer_id.count()

from sklearn import preprocessing
pre = preprocessing.LabelEncoder()
pre.fit(cate_customer_id)
nume_customer_id = pre.transform(cate_customer_id)

nume_customer_id

dataset["customer_id"] = nume_customer_id

dataset.head()

x = dataset.iloc[:,[0,14]]
y = dataset.iloc[:,17] ##coupon code

x.head()

y.head()

x.isnull()

x.isnull().sum()

x = x.interpolate()

x.isnull().sum()

y.isnull().sum()

from sklearn.model_selection import GridSearchCV

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

knn_2 = KNeighborsClassifier()

clf = GridSearchCV(knn_2, hyperparameters, cv=5)

best_model = clf.fit(x,y)

print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

classifier = KNeighborsClassifier(n_neighbors=28, leaf_size=1, p=1, metric='minkowski', metric_params=None, n_jobs=None)

fit = classifier.fit(x,y)





test_data = pd.read_csv("Test.csv")

test_data

test_customer_id = test_data["customer_id"].copy()

test_customer_id.count()

from sklearn import preprocessing
pre = preprocessing.LabelEncoder()
pre.fit(test_customer_id)
test_nume_customer_id = pre.transform(test_customer_id)

test_data["customer_id"] = test_nume_customer_id

test_data.head()

x_test = test_data.iloc[:,[0,14]]

x_test.isnull().sum()

x_test = x_test.interpolate()

x_test.isnull().sum()



"""Coupon code predictions"""

y_pred_coupon = classifier.predict(x_test)

y_pred_coupon





x2 = dataset.iloc[:,[0,16]]
y2 = dataset.iloc[:,18]  ## payment code

x2.head()

y2.head()

fit = classifier.fit(x2,y2)

y_pred_payment_code = classifier.predict(x_test)



"""Payment code prediction"""

y_pred_payment_code











