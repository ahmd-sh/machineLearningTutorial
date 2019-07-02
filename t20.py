#Similar code as in t14.py

import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array(([6,2,1,2,2,3,4,2,3],[8,4,3,5,2,3,1,3,2]))
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)