import os
import pandas as pd
from sklearn import tree

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
dt.fit(X_train,y_train)

#predict the outcome using decision tree
titanic_t = pd.read_csv("train.csv")

titanic_test_mini = titanic_t[['Pclass', 'Sex', 'Embarked']]
titanic_test_dummy = pd.get_dummies(titanic_test_mini, columns=['Pclass', 'Sex', 'Embarked'])

titanic_test_dummy.head()
titanic_test_dummy.info


#X_test = titanic_test_dummy
titanic_t['Survived'] = dt.predict(X_train)
titanic_t.to_csv("submission_overfit.csv", columns=['PassengerId','Survived'], index=False)