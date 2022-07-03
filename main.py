import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('train.csv')

logr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=0)
svm = svm.SVC()
mlp = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0)
gbc = GradientBoostingClassifier(n_estimators=10)

data.drop(['sr','ID','Description','Location Description','Date','Block','Location','FBI Code','Case Number','X Coordinate','Y Coordinate','Community Area','Updated On','IUCR'], inplace=True, axis=1)


nan_value = float("NaN")
data.replace("", nan_value, inplace=True)
data.dropna(subset = ['Ward','District','Latitude','Longitude'], inplace=True)
data.drop_duplicates()

le = LabelEncoder()
data['Arrest'] = le.fit_transform(data['Arrest'])
data['Primary Type'] = le.fit_transform(data['Primary Type'])

x = data.drop('Arrest',axis=1)
y = data['Arrest']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3 )

logr.fit(x_train, y_train)
rfc.fit(x_train, y_train)
dtc.fit(x_train, y_train)
svm.fit(x_train, y_train)
mlp.fit(x_train, y_train)
gbc.fit(x_train, y_train)

ylogr_predict = logr.predict(x_test)
rfcy_predict = rfc.predict(x_test)
dtcy_predict = dtc.predict(x_test)
svmy_predict = svm.predict(x_test)
mlpy_predict = mlp.predict(x_test)
gbcy_predict = gbc.predict(x_test)

print('Logistic:', accuracy_score(y_test, ylogr_predict))
print('Random Forest:', accuracy_score(y_test, rfcy_predict))
print('Decision Tree:', accuracy_score(y_test, dtcy_predict))
print('Support Vector:', accuracy_score(y_test, svmy_predict))
print('MLP:', accuracy_score(y_test,  mlpy_predict))
print('Gradient Boosting:', accuracy_score(y_test,  gbcy_predict))

'''
Accuracy Score:
Logistic: 0.7448591012947449
Random Forest: 0.8347296268088348
Decision Tree: 0.7928408225437928
Support Vector: 0.7448591012947449
MLP: 0.7448591012947449
Gradient Boosting: 0.853008377760853
'''
