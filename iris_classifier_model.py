# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os

# %%
iris_data = datasets.load_iris()
print(iris_data)

# %%
X=iris_data.data
X

# %%
Y=iris_data.target
Y

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# %%
lin_reg_model = LinearRegression()
log_reg_model = LogisticRegression()
svm_model = SVC()

lin_reg_model = lin_reg_model.fit(X_train,Y_train)
log_reg_model = log_reg_model.fit(X_train,Y_train)
svm_model = svm_model.fit(X_train,Y_train)

print(lin_reg_model.score(X_test,Y_test))
print(log_reg_model.score(X_test,Y_test))
print(svm_model.score(X_test,Y_test))

# %%
DIR = 'C:\\Users\\Asus\\Desktop\\machine_learning\\iris classifier\\'
pickle.dump(lin_reg_model,open(os.path.join(DIR+'lin_reg_model.pkl'),'wb'))
pickle.dump(log_reg_model,open(os.path.join(DIR+'log_reg_model.pkl'),'wb'))
pickle.dump(svm_model,open(os.path.join(DIR+'svm_model.pkl'),'wb'))
