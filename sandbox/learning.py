from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge as ridge_regression
import os, pickle

os.chdir("/home/cemarks/Projects/cancer/sandbox")

with open("ml_dataframe.pkl",'rb') as f:
    X = pickle.load(f)




predictor_columns = X.columns[4:14]
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X[predictor_columns])
lin = ridge_regression(alpha=1)
lin.fit(X[predictor_columns],X['metric2'])
s = lin.score(X[predictor_columns],X['metric2'])


lin = ridge_regression(alpha=10)
lin.fit(X_poly,X['metric2'])
s = lin.score(X_poly,X['metric2'])
print(s)