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

TRAINING_SET_FRAC = 0.98
# WORKING_DIR = "/home"
# MODEL_DIR = WORKING_DIR

# Load data
with open(os.path.join(WORKING_DIR,"expanded_training_X.pkl"),'rb') as f:
    X = pickle.load(f)

X.insert(
    X.shape[1],
    'db_weight',
    [0] * X.shape[0]
)

col_inds = (X.columns == 'db_weight')

X.loc[X['DB'] == 'REMBRANDT-leaderboard',col_inds] = 25
X.loc[X['DB'] == 'Outcome-Predictors-leaderboard',col_inds] = 50
X.loc[X['DB'] == 'APOLLO-2-leaderboard',col_inds] = 44
X.loc[X['DB'] == 'ROI-Masks-leaderboard',col_inds] = 23


# Get best value columns

# Only train on columns that have matches
X_VALUE_REGRESS = X.loc[X['metric2_max'] > 0]

def rr_transform(x):
    predictor_columns = [
        "secondary_search",
        "ftsearch_cde",
        # "ftsearch_dec",
        # "syn_classsum",
        # "syn_propsum",
        # "syn_objsum",
        # "syn_classmax",
        # "syn_propmax",
        # "syn_objmax",
        # "ftsearch_question",
        "enum_concept_search",
        # "enum_answer_search",
        # "answer_count_score",
        # "value_score",
        # "max_cde",
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
        # "pct_cde",
        # "pct_dec",
        "pct_que",
        # "pct_syn_classsum",
        # "pct_syn_propsum",
        # "pct_syn_objsum",
        # "pct_syn_classmax",
        # "pct_syn_propmax",
        # "pct_syn_objmax",
        # "pct_enum_concept",
        # "pct_enum_ans",
        # "pct_ans_score",
        # "pct_val_score",
        # "pct_secondary_search",
        # "cde_frac",
        # "dec_frac",
        # "que_frac",
        # "syn_classsum_frac",
        # "syn_propsum_frac",
        # "syn_objsum_frac",
        # "syn_classmax_frac",
        # "syn_propmax_frac",
        # "syn_objmax_frac",
        # "enum_concept_frac",
        # "enum_ans_frac",
        # "ans_score_frac",
        # "val_score_frac",
        "cde_norm",
        # "dec_norm",
        # "syn_classsum_norm",
        # "syn_propsum_norm",
        # "syn_objsum_norm",
        # "syn_classmax_norm",
        # "syn_propmax_norm",
        "syn_objmax_norm",
        "question_norm",
        # "enum_concept_norm",
        "enum_ans_norm",
        # "ans_score_norm",
        "val_score_norm",
        # "n",
        "logn"
    ]
    poly = PolynomialFeatures(degree = 2)
    Z_poly = poly.fit_transform(x[predictor_columns])
    # Z_poly = x[predictor_columns]
    return Z_poly


BEST_VALUE = 0.01

X_VALUE_REGRESS.insert(
    X_VALUE_REGRESS.shape[1],
    'Y',
    (X_VALUE_REGRESS['metric2']+X_VALUE_REGRESS['metric1']*BEST_VALUE/1.5)/(1+BEST_VALUE/1.5)
)
# X_VALUE_REGRESS['Y'] = (X_VALUE_REGRESS['metric2_frac'] > 0.8).astype('int')

# Separate into training & test
unique_columns=X_VALUE_REGRESS[['DB','col_no']].drop_duplicates().values
rand_ints = np.random.permutation(range(len(unique_columns)))

train_test_splitpoint = int(TRAINING_SET_FRAC*len(rand_ints))
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



weights = X_VALUE_REGRESS['db_weight'].loc[train_vector] / 142 * 1/X_VALUE_REGRESS['logn'].loc[train_vector]

BASE = 8
TANSHIFT = 2/3
EXP = 50


YY = (EXP ** X_VALUE_REGRESS['Y'].loc[train_vector] - 1)/(EXP-1)
YT = (EXP ** X_VALUE_REGRESS['Y'].loc[test_vector] - 1)/(EXP-1)
# YY = np.log((BASE-1)*X_VALUE_REGRESS['metric2_frac'].loc[train_vector]+1)/np.log(BASE)
# YT = np.log((BASE-1)*X_VALUE_REGRESS['metric2_frac'].loc[test_vector]+1)/np.log(BASE)
# YY = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['metric2_frac'].loc[train_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YT = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['metric2_frac'].loc[test_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YY = X_VALUE_REGRESS['metric2_frac'].loc[train_vector]
# YT = X_VALUE_REGRESS['metric2_frac'].loc[test_vector]
# YY = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['Y'].loc[train_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YT = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['Y'].loc[test_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YY = X_VALUE_REGRESS['Y'].loc[train_vector]
# YT = X_VALUE_REGRESS['Y'].loc[test_vector]
# YY = np.log((BASE-1)*X_VALUE_REGRESS['Y'].loc[train_vector]+1)/np.log(BASE)
# YT = np.log((BASE-1)*X_VALUE_REGRESS['Y'].loc[test_vector]+1)/np.log(BASE)

RANGE = range(-6,3)
test_scores = []
for k in RANGE:
    rfr = Ridge(
        alpha=10**k,
        fit_intercept = True,
        normalize= True,
        tol = 0.001,
        solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
    )
    rfr.fit(XX,YY,sample_weight=weights)
    print(k)
    print(rfr.score(XX,YY))
    print(rfr.score(XT,YT))
    TMP_DF = X_VALUE_REGRESS.loc[test_vector]
    TMP_DF.insert(
        TMP_DF.shape[1],
        'Yhat',
        rfr.predict(
            rr_transform(TMP_DF)
        )
    )
    metrics = []
    for i in range(len(test_inds)):
        tsv = unique_columns[test_inds[i]][0]
        col_no = unique_columns[test_inds[i]][1]
        display_df = TMP_DF.loc[(TMP_DF['DB']==tsv) & (TMP_DF['col_no']==col_no),['metric1','metric2','Yhat','index']]
        display_list = display_df.values.tolist()
        display_list.sort(key = lambda z: z[2], reverse=True)
        metrics.append(
            np.max(
                [
                    (1 + (2-i)/1.5)*display_list[0][0] + (3-i)*display_list[0][1] for i in range(3)
                ]
            )
        )
    ts = np.mean(metrics)
    print("SCORE: {0:1.3f}".format(ts))
    test_scores.append(ts)
    print()


best_alpha = list(RANGE)[np.argmax(test_scores)]
## Submission version: over-regularize
best_alpha = -0.5
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
rfr.fit(XX,YY,sample_weight=weights)
print(rfr.score(XX,YY))
print(rfr.score(XT,YT))
print()

# Save to model dir
model_dict = {
    'model': rfr
}
with open(os.path.join(MODEL_DIR,'value_regression.pkl'), 'wb') as f:
    pickle.dump(model_dict,f)


## Print ordering of test set results.

if 'metric2_predict' in X_VALUE_REGRESS.columns:
    X_VALUE_REGRESS['metric2_predict'] = rfr.predict(
        rr_transform(X_VALUE_REGRESS)
    )
else:
    X_VALUE_REGRESS.insert(
        X_VALUE_REGRESS.shape[1],
        'metric2_predict',
        rfr.predict(
            rr_transform(X_VALUE_REGRESS)
        )
    )  

metrics = []
for i in range(len(test_inds)):
    tsv = unique_columns[test_inds[i]][0]
    col_no = unique_columns[test_inds[i]][1]
    display_df = X_VALUE_REGRESS.loc[(X_VALUE_REGRESS['DB']==tsv) & (X_VALUE_REGRESS['col_no']==col_no),['metric1','metric2','metric2_predict','index']]
    display_list = display_df.values.tolist()
    display_list.sort(key = lambda z: z[2], reverse=True)
    metrics.append(
        np.max(
            [
                (1 + (2-i)/1.5)*display_list[0][0] + (3-i)*display_list[0][1] for i in range(3)
            ]
        )
    )
    # for ii in range(min(20,len(display_list))):
    #     print(display_list[ii])
    # print()
    # print()

print("SCORE: {0:1.3f}".format(np.mean(metrics)))
