import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("c:/__backup/kaggle/Titanic/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train_mini = titanic_train[['Pclass', 'Sex', 'Embarked', 'Survived']]

titanic_train_dummy = pd.get_dummies(titanic_train_mini, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train_dummy.shape
titanic_train_dummy.info()
titanic_train_dummy.head()

X_train = titanic_train_dummy.drop(['Survived'], 1)
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
cv_scores = model_selection.cross_val_score(dt, X_train, y_train, cv = 10)
print(cv_scores)
print(cv_scores.mean())
dt.fit(X_train, y_train)

dt.score(X_train, y_train)

y_pred = dt.predict(X_train)
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
metrics.accuracy_score(y_train, y_pred)