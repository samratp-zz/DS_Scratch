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

param_grid = {'max_depth':[2,4,6,8,10,15,20,30,50], 'min_samples_split':[2,3,5,8,10,15,20,30,50]}
dt_grid = model_selection.GridSearchCV(dt, param_grid, n_jobs = 5, cv = 10)
dt_grid.fit(X_train, y_train)
dt_grid.grid_scores_
dt_grid.best_score_
dt_grid.score(X_train, y_train)

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")

titanic_test_mini = titanic_test[['Pclass', 'Sex', 'Embarked']]
titanic_test_dummy = pd.get_dummies(titanic_test_mini, columns=['Pclass', 'Sex', 'Embarked'])

titanic_test_dummy.head()
titanic_test_dummy.info


X_test = titanic_test_dummy
titanic_test['Survived'] = dt_grid.predict(X_test)
titanic_test.to_csv("submission_tuned.csv", columns=['PassengerId','Survived'], index=False)