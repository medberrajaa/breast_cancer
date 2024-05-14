from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

breast_cancer = load_breast_cancer()
X = breast_cancer.data
X = np.array(X)
Y = breast_cancer.target
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(breast_cancer.target_names)

svm = SVC(kernel="linear")
svm = svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print(score)
dump(svm, "svm_model")

