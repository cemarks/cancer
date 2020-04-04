from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge as ridge_regression
from sklearn.linear_model import LogisticRegression as logistic_regression
import os, pickle
import numpy as np

os.chdir("/home/cemarks/Projects/cancer/sandbox")

with open("ml_dataframe.pkl",'rb') as f:
    X = pickle.load(f)


X['max_cde'] = 0
X['max_dec'] = 0
X['max_que'] = 0
X['max_syn'] = 0
X['max_enum_concept'] = 0
X['max_enum_ans'] = 0


for i in X['DB'].unique():
    for j in X['col_no'].loc[X['DB'] == i].unique():
        inds = (X['DB'] == i) & (X['col_no'] == j)
        X.loc[inds,'max_cde'] = (1+max(X['ftsearch_cde'].loc[inds]))
        X.loc[inds,'max_dec'] = (1+max(X['ftsearch_dec'].loc[inds]))
        X.loc[inds,'max_que'] = (1+max(X['ftsearch_question'].loc[inds]))
        X.loc[inds,'max_syn'] = (1+max([max(X['ftsearch_syn_prop'].loc[inds]),max(X['ftsearch_syn_obj'].loc[inds])]))
        X.loc[inds,'max_enum_concept'] = (1+max(X['enum_concept_search'].loc[inds]))
        X.loc[inds,'max_enum_ans'] = (1+max(X['enum_answer_search'].loc[inds]))

print("\n".join([str(i) + " " + X.columns[i] for i in range(X.shape[1])]))
X['log_cde'] = X.apply(lambda z: z[7]/np.log(1+z[16]),axis=1)

X['metric3'] = X.apply(lambda z: 1 if z[14] > 1 else 0,axis=1)
X['metric4'] = X.apply(lambda z: 1 if z[15] > 0 else 0,axis=1)

predictor_columns = [
#    "ftsearch_syn_class",
    "ftsearch_syn_prop",
    "ftsearch_syn_obj",
    "ftsearch_cde",
    "ftsearch_dec",
    "ftsearch_question",
    "enum_concept_search",
    "enum_answer_search",
    "answer_count_score",
    "value_score",
#    "log_cde",
    "max_cde",
    "max_dec",
    "max_que",
    "max_syn",
    "max_enum_concept",
    "max_enum_ans"
]

# predictor_columns = X.columns[4:14]
lin = ridge_regression(alpha=1)
lin.fit(X[predictor_columns],X['metric2'])
s = lin.score(X[predictor_columns],X['metric2'])


poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X[predictor_columns])
lin = ridge_regression(alpha=10)
lin.fit(X_poly,X['metric2'])
s = lin.score(X_poly,X['metric2'])
print(s)

logreg = logistic_regression(
    class_weight = {
        0:0.01,
        1:0.99
    },
    C = 0.1
)
logreg.fit(X_poly,X['metric3'])
s = logreg.score(X_poly,X['metric4'])
print(s)








