from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
Y = iris.target

iris_ss = ShuffleSplit(train_size = 0.5, test_size = 0.5, random_state = 0)
train_index, test_index = next(iris_ss.split(X))

X_train, Y_train = X[train_index], Y[train_index]
X_test, Y_test = X[test_index], Y[test_index]

clf = svm.SVC()
clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))


plt.scatter(X[:50, 0], X[:50, 1], color = 'r', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'g', marker = '+', label = 'versicolor')
plt.scatter(X[100:,0], X[100:,1], color = 'b', marker = 'x', label = 'virginica')
plt.title("Iris Plants Database")
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.legend()
plt.show()
