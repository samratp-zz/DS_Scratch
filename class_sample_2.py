import os
import pandas as pd
from sklearn import tree
import io
import pydot

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

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("dt4.pdf")

#predict the outcome using decision tree
titanic_test = pd.read_csv("test.csv")

titanic_test_mini = titanic_test[['Pclass', 'Sex', 'Embarked']]
titanic_test_dummy = pd.get_dummies(titanic_test_mini, columns=['Pclass', 'Sex', 'Embarked'])

titanic_test_dummy.head()
titanic_test_dummy.info


X_test = titanic_test_dummy
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
















