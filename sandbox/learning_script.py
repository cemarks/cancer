import pandas as pd
import os, pickle
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.model_selection import KFold
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from sklearn import svm
from matplotlib import pyplot as plt
from random import sample

WORKING_DIR = "/home/cemarks/Projects/cancer/sandbox"
MODEL_DIR = "/home/cemarks/Projects/cancer/sandbox"

# Load data
with open(os.path.join(WORKING_DIR,"expanded_training_X.pkl"),'rb') as f:
    X = pickle.load(f)

# Indices not in the caDSR export
not_founds = [
    (6002302, "APOLLO-2-leaderboard", 38),
    (2608243, "Outcome-Predictors-leaderboard", 31)
]

# Remove these rows from data
found_inds = (X['DB'] != not_founds[0][1]) | (X['col_no'] != not_founds[0][2])
for j in not_founds[1:len(not_founds)]:
    found_inds = found_inds & ((X['DB'] != j[1]) | (X['col_no'] != j[2]))
X = X.loc[found_inds]

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
for train_index,test_index in INDS:
    # XX = X_NOMATCH[predictor_columns].iloc[train_index] 
    # XT = X_NOMATCH[predictor_columns].iloc[test_index] 
    XX = lr_transform(X_NOMATCH.iloc[train_index])
    XT = lr_transform(X_NOMATCH.iloc[test_index])
    YY = X_NOMATCH['Y'].iloc[train_index]
    YT = X_NOMATCH['Y'].iloc[test_index]
    s = []
    st = []
    for c in range(-8,8):
        logreg = LogisticRegression(
            C = 10**c,
            max_iter = 10000,
            tol=0.00001,
            solver='lbfgs' # liblinear, lbfgs, newton-cg, sag, saga
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
best_c = list(range(-8,8))[mt.argmax()]
best_model = LogisticRegression(
    C = 10 ** best_c,
    max_iter = 10000,
    tol=0.00001,
    solver='newton-cg'
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

# Plot ROC curves
probs = best_model.predict_proba(
    XX
)

plot_roc(probs,YY,0.6)

probs = logreg.predict_proba(
    XT
)

plot_roc(probs,YT,0.5)

model_dict = {
    'model': best_model
}

with open(os.path.join(MODEL_DIR,'nomatch_model.pkl'), 'wb') as f:
    pickle.dump(model_dict,f)


#################

# Round II: get best value columns

# Only train on columns that have matches
X_VALUE_REGRESS = X.loc[X['metric4']==1]

# Separate into training & test
unique_columns=X_VALUE_REGRESS[['DB','col_no']].drop_duplicates().values
rand_ints = np.random.permutation(range(len(unique_columns)))
train_test_splitpoint = int(0.85*len(rand_ints))
training_inds = rand_ints[0:train_test_splitpoint]
test_inds = rand_ints[train_test_splitpoint:len(rand_ints)]

train_vector = pd.Series([False]*len(X_VALUE_REGRESS))
train_vector.index = X_VALUE_REGRESS.index

for t,i in enumerate(training_inds):
    train_vector = train_vector | ((X_VALUE_REGRESS['DB']==unique_columns[i][0]) & (X_VALUE_REGRESS['col_no']==unique_columns[i][1]))

test_vector = pd.Series([False]*len(X_VALUE_REGRESS))
test_vector.index = X_VALUE_REGRESS.index

for t,i in enumerate(test_inds):
    test_vector = test_vector | ((X_VALUE_REGRESS['DB']==unique_columns[i][0]) & (X_VALUE_REGRESS['col_no']==unique_columns[i][1]))


XX = rr_transform(X_VALUE_REGRESS.loc[train_vector])
XT = rr_transform(X_VALUE_REGRESS.loc[test_vector])

YY = X_VALUE_REGRESS['metric2_frac'].loc[train_vector].pow(2)
YT = X_VALUE_REGRESS['metric2_frac'].loc[test_vector].pow(2)

# Fit model
test_scores = []
for k in range(-4,4):
    rfr = Ridge(
        alpha=10**k,
        fit_intercept = True,
        normalize= True,
        tol = 0.00001,
        solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
    )
    rfr.fit(XX,YY)
    print(k)
    print(rfr.score(XX,YY))
    ts = rfr.score(XT,YT)
    print(ts)
    print()
    test_scores.append(ts)

best_alpha = list(range(-4,4))[np.argmax(test_scores)]
rfr = Ridge(
    alpha=10 ** best_alpha,
    fit_intercept = True,
    normalize= True,
    tol = 0.00001,
    solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
)
# rfr.fit(Z[predictor_columns].loc[train_vector],(Z['metric2'].loc[train_vector]))
# print(k)
# print(rfr.score(Z[predictor_columns].loc[train_vector],(Z['metric2'].loc[train_vector])))
# print(rfr.score(Z[predictor_columns].loc[test_vector],(Z['metric2'].loc[test_vector])))
rfr.fit(XX,YY)
print(rfr.score(XX,YY))
print(rfr.score(XT,YT))
print()

# Save to model dir
model_dict = {
    'model': rfr
}
with open(os.path.join(MODEL_DIR,'value_regression.pkl'), 'wb') as f:
    pickle.dump(model_dict,f)


# Print ordering of test set results.
X_VALUE_REGRESS['metric2_predict'] = rfr.predict(
    rr_transform(X_VALUE_REGRESS)
)   
for i in range(12):
    tsv = unique_columns[test_inds[i]][0]
    col_no = unique_columns[test_inds[i]][1]
    display_df = X_VALUE_REGRESS.loc[(X_VALUE_REGRESS['DB']==tsv) & (X_VALUE_REGRESS['col_no']==col_no),['metric1','metric2','metric2_predict','index']]
    display_list = display_df.values.tolist()
    display_list.sort(key = lambda z: z[2], reverse=True)
    for ii in range(min(20,len(display_list))):
        print(display_list[ii])
    print()
    print()

