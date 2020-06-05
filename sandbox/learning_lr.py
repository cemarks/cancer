import pandas as pd
import os, pickle
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression,Lasso
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from sklearn import svm
from random import sample

from my_modules.learning_models import *

WORKING_DIR = "/home/cemarks/Projects/cancer/sandbox"
MODEL_DIR = "/home/cemarks/Projects/cancer/sandbox"

# Load data
with open(os.path.join(WORKING_DIR,"expanded_training_X.pkl"),'rb') as f:
    X = pickle.load(f)

# Create dataframe for nomatch classification
nomatch_data = []
for i in X['DB'].unique():
    for j in X['col_no'].loc[X['DB'] == i].unique():
        nomatch_data.append(X.loc[(X['DB']==i) & (X['col_no'] == j)].iloc[0])

X_NOMATCH = pd.DataFrame(nomatch_data)
X_NOMATCH['Y'] = (X_NOMATCH['metric2_max'] == 0).astype('int') #These are nomatch rows.

# Do nomatch classification
kf = KFold(n_splits=10,shuffle=True)
INDS = kf.split(X_NOMATCH)
o = []
ot = []

SOLVER = 'liblinear'
TOL = 0.0001
MAX_ITER = 10000
RANGE = range(0,10)
PENALTY = 'l1'

for train_index,test_index in INDS:
    # XX = X_NOMATCH[predictor_columns].iloc[train_index] 
    # XT = X_NOMATCH[predictor_columns].iloc[test_index] 
    XX = lr_transform(X_NOMATCH.iloc[train_index])
    XT = lr_transform(X_NOMATCH.iloc[test_index])
    YY = X_NOMATCH['Y'].iloc[train_index]
    YT = X_NOMATCH['Y'].iloc[test_index]
    s = []
    st = []
    for c in RANGE:
        logreg = LogisticRegression(
            C = 10**c,
            penalty = PENALTY,
            max_iter = MAX_ITER,
            tol = TOL,
            solver = SOLVER # liblinear, lbfgs, newton-cg, sag, saga
        )
        logreg.fit(XX,YY)
        s.append(logreg.score(XX,YY))
        st.append(logreg.score(XT,YT))
    o.append(s)
    ot.append(st)

# View results
o = np.array(o)
m = np.mean(o,axis=0)
print(m)
print()
ot = np.array(ot)
mt = np.mean(ot,axis=0)
print(mt)
print()

# Refit best model
best_c = list(RANGE)[mt.argmax()]
## Submission version: overregularize
best_c = 1
best_model = LogisticRegression(
    C = 10 ** best_c,
    penalty = PENALTY,
    max_iter = MAX_ITER,
    tol=TOL,
    solver = SOLVER # liblinear, lbfgs (default), newton-cg, sag, saga
)
test_inds = sample(range(X_NOMATCH.shape[0]),10)
tng_inds = [i for i in range(X_NOMATCH.shape[0]) if i not in test_inds]

XX = lr_transform(X_NOMATCH.iloc[tng_inds])
XT = lr_transform(X_NOMATCH.iloc[test_inds])

YY = X_NOMATCH['Y'].iloc[tng_inds]
YT = X_NOMATCH['Y'].iloc[test_inds]
best_model.fit(XX,YY)
print(best_model.score(XX,YY))
print(best_model.score(XT,YT))


XR = lr_transform(X_NOMATCH.loc[X_NOMATCH['DB']=='REMBRANDT-leaderboard'])
YR = X_NOMATCH['Y'].loc[X_NOMATCH['DB']=='REMBRANDT-leaderboard']
XA = lr_transform(X_NOMATCH.loc[X_NOMATCH['DB']=='APOLLO-2-leaderboard'])
YA = X_NOMATCH['Y'].loc[X_NOMATCH['DB']=='APOLLO-2-leaderboard']
print(best_model.score(XA,YA))

print(best_model.score(XR,YR))

# Plot ROC curves
# probs = best_model.predict_proba(
#     XX
# )

# plot_roc(probs,YY,0.6)

# probs = logreg.predict_proba(
#     XT
# )

# plot_roc(probs,YT,0.5)

model_dict = {
    'model': best_model
}

with open(os.path.join(MODEL_DIR,'nomatch_model.pkl'), 'wb') as f:
    pickle.dump(model_dict,f)


