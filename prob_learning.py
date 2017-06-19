import os
import pandas as pd
from sklearn import naive_bayes

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("c:/__backup/kaggle/Titanic/")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

nb_estimator = naive_bayes.GaussianNB()
nb_estimator.fit(X_train, y_train)


titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = nb_estimator.predict(X_test)
titanic_test.to_csv("submission_GNB.csv", columns=['PassengerId','Survived'], index=False)