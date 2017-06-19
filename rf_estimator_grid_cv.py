import os
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection

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

rf_tree_estimator = ensemble.RandomForestClassifier()
param_grid = {'n_estimators' : [100,500,1000], 'max_depth':[2,10,50], 'min_samples_split':[2,15,50], 'max_features':[4,8]}
rf_grid = model_selection.GridSearchCV(rf_tree_estimator, param_grid, n_jobs = 15, cv = 10)
rf_grid.fit(X_train, y_train)
rf_grid.grid_scores_
rf_grid.best_score_
rf_grid.score(X_train, y_train)


titanic_test = pd.read_csv("test.csv")
titanic_test.shape
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()

X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
titanic_test['Survived'] = rf_grid.predict(X_test)
titanic_test.to_csv("submission_rf_grid_cv.csv", columns=['PassengerId','Survived'], index=False)