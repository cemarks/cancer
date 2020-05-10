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


XR = lr_transform(X_NOMATCH.loc[X_NOMATCH['DB']=='REMBRANDT-leaderboard'])
YR = X_NOMATCH['Y'].loc[X_NOMATCH['DB']=='REMBRANDT-leaderboard']
XA = lr_transform(X_NOMATCH.loc[X_NOMATCH['DB']=='APOLLO-2-leaderboard'])
YA = X_NOMATCH['Y'].loc[X_NOMATCH['DB']=='APOLLO-2-leaderboard']
print(best_model.score(XA,YA))

print(best_model.score(XR,YR))

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
X_VALUE_REGRESS = X.loc[X['metric2_max'] > 0]

X_VALUE_REGRESS['Rembrandt'] = (X_VALUE_REGRESS['DB'] == 'REMBRANDT-leaderboard').astype('int')
X_VALUE_REGRESS['Apollo'] = (X_VALUE_REGRESS['DB'] == 'APOLLO-2-leaderboard').astype('int')
X_VALUE_REGRESS['Outcome'] = (X_VALUE_REGRESS['DB'] == 'Outcome-Predictors-leaderboard').astype('int')
X_VALUE_REGRESS['ROI'] = (X_VALUE_REGRESS['DB'] == 'ROI-Masks-leaderboard').astype('int')

# Separate into training & test
unique_columns=X_VALUE_REGRESS[['DB','col_no']].drop_duplicates().values
rand_ints = np.random.permutation(range(len(unique_columns)))
train_test_splitpoint = int(0.75*len(rand_ints))
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



def rr_transform(x):
    predictor_columns = [
        # "secondary_search",
        "ftsearch_cde",
        # "ftsearch_dec",
        # "syn_classsum",
        "syn_propsum",
        # "syn_objsum",
        # "syn_classmax",
        # "syn_propmax",
        # "syn_objmax",
        # "ftsearch_question",
        "enum_concept_search",
        # "enum_answer_search",
        # "answer_count_score",
        # "value_score",
        "max_cde",
        # "max_dec",
        # "max_que",
        # "max_syn_classsum",
        # "max_syn_propsum",
        # "max_syn_objsum",
        # "max_syn_classmax",
        # "max_syn_propmax",
        # "max_syn_objmax",
        # "max_enum_concept",
        # "max_enum_ans",
        # "max_ans_score",
        # "max_val_score",
        # "max_secondary_search",
        "pct_cde",
        # "pct_dec",
        # "pct_que",
        # "pct_syn_classsum",
        "pct_syn_propsum",
        # "pct_syn_objsum",
        # "pct_syn_classmax",
        # "pct_syn_propmax",
        # "pct_syn_objmax",
        "pct_enum_concept",
        # "pct_enum_ans",
        # "pct_ans_score",
        # "pct_val_score",
        # "pct_secondary_search",
        "cde_frac",
        "dec_frac",
        "que_frac",
        # "syn_classsum_frac",
        # "syn_propsum_frac",
        "syn_objsum_frac",
        # "syn_classmax_frac",
        "syn_propmax_frac",
        # "syn_objmax_frac",
        # "enum_concept_frac",
        "enum_ans_frac",
        "ans_score_frac",
        "val_score_frac",
        # "n",
        "logn"
    ]
    # poly = PolynomialFeatures(degree = 2)
    # Z_poly = poly.fit_transform(x[predictor_columns])
    Z_poly = x[predictor_columns]
    return Z_poly




XX = rr_transform(X_VALUE_REGRESS.loc[train_vector])
XT = rr_transform(X_VALUE_REGRESS.loc[test_vector])


from sklearn.decomposition import PCA
pca = PCA(n_components = XX.shape[1])
XX = pca.fit_transform(XX)
XT = pca.transform(XT)


YY = (100**(X_VALUE_REGRESS['metric2_frac'].loc[train_vector])-1)/(100-1)
YT = (100**(X_VALUE_REGRESS['metric2_frac'].loc[test_vector])-1)/(100-1)


cols = range(0,12)
# Fit model
test_scores = []
for k in range(-5,3):
    rfr = Ridge(
        alpha=10**k,
        fit_intercept = True,
        normalize= True,
        tol = 0.001,
        solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
    )
    rfr.fit(XX[:,cols],YY)
    print(k)
    print(rfr.score(XX[:,cols],YY))
    ts = rfr.score(XT[:,cols],YT)
    print(ts)
    print()
    test_scores.append(ts)


# test_scores = []
# for k in range(-5,3):
#     rfr = Lasso(
#         alpha=10**(k-6),
#         fit_intercept = True,
#         normalize= True,
#         tol = 0.00001,
#     )
#     rfr.fit(XX,YY)
#     print(k)
#     print(rfr.score(XX,YY))
#     ts = rfr.score(XT,YT)
#     print(ts)
#     print()
#     test_scores.append(ts)



# test_scores = []
# for k in [3,5,8,12]:
#     rfr = RandomForestRegressor(
#         n_estimators = 80,
#         max_features = k, # auto,sqrt,log2, [float], or [int]
#         max_samples = 12000
#     )
#     rfr.fit(XX,YY)
#     print(k)
#     print(rfr.score(XX,YY))
#     ts = rfr.score(XT,YT)
#     print(ts)
#     print()
#     test_scores.append(ts)



best_alpha = list(range(-5,3))[np.argmax(test_scores)]
rfr = Ridge(
    alpha=10 ** best_alpha,
    fit_intercept = True,
    normalize= True,
    tol = 0.00001,
    solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
)
# best_alpha = list(range(-2,3))[np.argmax(test_scores)]
# rfr = Lasso(
#     alpha=10 ** (best_alpha-6),
#     fit_intercept = True,
#     normalize= True,
#     tol = 0.00001,
# )
# rfr = RandomForestRegressor(
#     n_estimators = 200,
#     max_features = 5, # auto,sqrt,log2, [float], or [int]
#     max_samples = 12000
# )
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
for i in range(len(test_inds)):
    tsv = unique_columns[test_inds[i]][0]
    col_no = unique_columns[test_inds[i]][1]
    display_df = X_VALUE_REGRESS.loc[(X_VALUE_REGRESS['DB']==tsv) & (X_VALUE_REGRESS['col_no']==col_no),['metric1','metric2','metric2_predict','index']]
    display_list = display_df.values.tolist()
    display_list.sort(key = lambda z: z[2], reverse=True)
    for ii in range(min(20,len(display_list))):
        print(display_list[ii])
    print()
    print()

