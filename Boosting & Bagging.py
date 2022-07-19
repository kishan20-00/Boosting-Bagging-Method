# Load Library

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


# Step1: Create the data set
X,y = make_moons(n_samples=10000, noise=.5, random_state=0)

# Step2: Split the training test set
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Step3: Fit a Decision Tree model as Comparison
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy_score(Y_test, y_pred)

# Step4: Fit a Random Forest Model, compared to Decision Tree
clf = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=0)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy_score(Y_test, y_pred)

# Step5: Fit a AdaBoost Model, comparing to Decision Tree
clf= AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy_score(Y_test, y_pred)

#Step6: Fit a Gradient Boosting Model, comparing to Decision Tree 
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy_score(Y_test, y_pred)
