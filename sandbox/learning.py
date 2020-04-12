import pandas as pd
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
import os, pickle
import numpy as np
from matplotlib import pyplot as plt
from random import sample

os.chdir("/home/cemarks/Projects/cancer/sandbox")

with open("expanded_training_X.pkl",'rb') as f:
    X = pickle.load(f)


# print("\n".join([str(i) + " " + X.columns[i] for i in range(X.shape[1])]))
# # X['log_cde'] = X.apply(lambda z: z[7]/np.log(1+z[16]),axis=1)

not_founds = [
    (6002302, "APOLLO-2-leaderboard", 38),
    (2608243, "Outcome-Predictors-leaderboard", 31)
]

INPUT_DIR = "/home/cemarks/Projects/cancer/data/leaderboard"
file_names = os.listdir(INPUT_DIR)
file_splits = [os.path.splitext(f) for f in file_names]
tsvs = [i for i in file_splits if i[1] == '.tsv']
dbs = [i[0] for i in tsvs]
found_inds = (X['DB'] != not_founds[0][1]) | (X['col_no'] != not_founds[0][2])
for j in not_founds[1:len(not_founds)]:
    found_inds = found_inds & ((X['DB'] != j[1]) | (X['col_no'] != j[2]))


Y = X.loc[found_inds]



z = []
stat_columns = [
    "max_cde",
    "max_dec",
    "max_que",
    "max_syn_classsum",
    "max_syn_propsum",
    "max_syn_objsum",
    "max_syn_classmax",
    "max_syn_propmax",
    "max_syn_objmax",
    "max_enum_concept",
    "max_enum_ans",
    "max_ans_score",
    "max_val_score",
    "pct_cde",
    "pct_dec",
    "pct_que",
    "pct_syn_classsum",
    "pct_syn_propsum",
    "pct_syn_objsum",
    "pct_syn_classmax",
    "pct_syn_propmax",
    "pct_syn_objmax",
    "pct_enum_concept",
    "pct_enum_ans",
    "pct_ans_score",
    "pct_val_score",
    "logn",
    "n",
    "metric2_max"
]
for i in Y['DB'].unique():
    for j in Y['col_no'].loc[Y['DB'] == i].unique():
        z.append(Y.loc[(Y['DB']==i) & (Y['col_no'] == j),stat_columns].iloc[0])



predictor_columns = [
    "max_cde",
    "max_dec",
    "max_que",
    # "max_syn_classsum",
    # "max_syn_propsum",
    # "max_syn_objsum",
    # "max_syn_classmax",
    "max_syn_propmax",
    # "max_syn_objmax",
    # "max_enum_concept",
    # "max_enum_ans",
    # "max_ans_score",
    # "max_val_score",
    # "pct_cde",
    # "pct_dec",
    # "pct_que",
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
    # "n",
    "logn"
]


D = pd.DataFrame(z)
D['Y'] = (D['metric2_max'] == 0).astype('int') #These are nomatch rows.
D['logn'] = np.log(D['n'])

kf = KFold(n_splits=10,shuffle=True)
INDS = kf.split(D)

o = []
ot = []

# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(D[predictor_columns])

for train_index,test_index in INDS:
    XX = D[predictor_columns].iloc[train_index] 
    XT = D[predictor_columns].iloc[test_index] 
    # XX = X_poly[train_index]
    # XT = X_poly[test_index]
    YY = D['Y'].iloc[train_index]
    YT = D['Y'].iloc[test_index]
    s = []
    st = []
#    for c in [0.00005,0.0001,0.0002,0.0005,0.001,0.005,0.01]:
    for c in range(-8,8):
        logreg = LogisticRegression(
            C = 10**c,
            max_iter = 10000,
            tol=0.000000001,
            solver='liblinear'
        )
        logreg.fit(XX,YY)
        s.append(logreg.score(XX,YY))
        st.append(logreg.score(XT,YT))
    o.append(s)
    ot.append(st)

o = np.array(o)
m = np.mean(o,axis=0)
print(m)
print()
ot = np.array(ot)
mt = np.mean(ot,axis=0)
print(mt)
print()







best_c = 10
best_model = LogisticRegression(
    C = best_c,
    max_iter = 10000,
    tol=0.000000001,
    solver='liblinear'
)
test_inds = sample(range(D.shape[0]),10)
tng_inds = [i for i in range(D.shape[0]) if i not in test_inds]

XX = D[predictor_columns].iloc[tng_inds]
XT = D[predictor_columns].iloc[test_inds]

YY = D['Y'].iloc[tng_inds]
YT = D['Y'].iloc[test_inds]
best_model.fit(XX,YY)
print(best_model.score(XX,YY))
print(best_model.score(XT,YT))


probs = best_model.predict_proba(
    XX
)

gt = [(probs[i][0],YY.iloc[i]) for i in range(len(probs))]
gt.sort(key=lambda z: z[0])
x = [0]
y = [0]

pos_count = sum([i[1] for i in gt])
neg_count = len(gt) - pos_count

x_count = 0
y_count = 0
xpt = []
ypt = []
found = False
for i in gt:
    if i[1] == 0:
        x_count+=1
    else:
        y_count+=1
    if (not found) and (i[0] > 0.66):
        found = True
        xpt.append(x_count/neg_count)
        ypt.append(y_count/pos_count)
    x.append(x_count/neg_count)
    y.append(y_count/pos_count)

x.append(1)
y.append(1)

plt.plot(x,y)
plt.scatter(xpt,ypt,s=30,c='red')
plt.plot([0,1],[0,1],"--")
plt.show()
plt.clf()
plt.close()


probs = logreg.predict_proba(
    XT
)

gt = [(probs[i][0],YT.iloc[i]) for i in range(len(probs))]
gt.sort(key=lambda z: z[0])
x = [0]
y = [0]

pos_count = sum([i[1] for i in gt])
neg_count = len(gt) - pos_count

x_count = 0
y_count = 0
for i in gt:
    if i[1] == 0:
        x_count+=1
    else:
        y_count+=1
    x.append(x_count/neg_count)
    y.append(y_count/pos_count)

x.append(1)
y.append(1)

plt.plot(x,y)
plt.plot([0,1],[0,1],"--")
plt.show()
plt.clf()
plt.close()

model_dict = {
    'predictor_columns': predictor_columns,
    'model': best_model
}

with open('nomatch_model.pkl', 'wb') as f:
    pickle.dump(model_dict,f)




#################

# Round II: get best value columns

Z = Y.loc[Y['metric4']==1]

predictor_columns = [
    "ftsearch_cde",
    "ftsearch_dec",
    # "syn_classsum",
    "syn_propsum",
    # "syn_objsum",
    # "syn_classmax",
    # "syn_propmax",
    # "syn_objmax",
    "ftsearch_question",
    "enum_concept_search",
    "enum_answer_search",
    "answer_count_score",
    "value_score",
    "max_cde",
    "max_dec",
    "max_que",
    # "max_syn_classsum",
    "max_syn_propsum",
    # "max_syn_objsum",
    # "max_syn_classmax",
    # "max_syn_propmax",
    # "max_syn_objmax",
    "max_enum_concept",
    # "max_enum_ans",
    "max_ans_score",
    "max_val_score",
    "pct_cde",
    "pct_dec",
    # "pct_que",
    # "pct_syn_classsum",
    "pct_syn_propsum",
    "pct_syn_objsum",
    # "pct_syn_classmax",
    #"pct_syn_propmax",
    #"pct_syn_objmax",
    "pct_enum_concept",
    # "pct_enum_ans",
    "pct_ans_score",
    "pct_val_score",
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
    # "n",
    "logn"
]


o=[]
for i in range(len(tsvs)):
    tsv = tsvs[i][0]
    J = Z['col_no'].loc[Z['DB']==tsv].unique()
    for j in range(len(J)):
        o.append((tsv,J[j]))

rand_ints = np.random.permutation(range(len(o)))
train_test_splitpoint = int(0.85*len(rand_ints))
training_inds = rand_ints[0:train_test_splitpoint]
test_inds = rand_ints[train_test_splitpoint:len(rand_ints)]

train_vector = pd.Series([False]*len(Z))
train_vector.index = Z.index

for t,i in enumerate(training_inds):
    train_vector = train_vector | ((Z['DB']==o[i][0]) & (Z['col_no']==o[i][1]))

test_vector = pd.Series([False]*len(Z))
test_vector.index = Z.index

for t,i in enumerate(test_inds):
    test_vector = test_vector | ((Z['DB']==o[i][0]) & (Z['col_no']==o[i][1]))

poly = PolynomialFeatures(degree = 2)
Z_poly= poly.fit_transform(Z[predictor_columns])

# for k in range(1,min(11,len(predictor_columns))):
for k in range(-4,4,1):
    # rfr = RandomForestRegressor(
    #     n_estimators = 30,
    #     max_features = k
    # )
    rfr = Ridge(
        alpha=10**k,
        fit_intercept = True,
        normalize= True,
        tol = 0.00001,
        solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
    )
    rfr.fit(Z_poly[train_vector],(Z['metric2_frac'].loc[train_vector]))
    print(k)
    print(rfr.score(Z_poly[train_vector],(Z['metric2_frac'].loc[train_vector])))
    print(rfr.score(poly.transform(Z[predictor_columns].loc[test_vector]),(Z['metric2_frac'].loc[test_vector])))
    print()

rfr = Ridge(
    alpha=0.1,
    fit_intercept = True,
    normalize= True,
    tol = 0.00001,
    solver='lsqr', # auto, svd, cholesky, lsqr, sparse_cg, sag, saga
)
# rfr.fit(Z[predictor_columns].loc[train_vector],(Z['metric2'].loc[train_vector]))
# print(k)
# print(rfr.score(Z[predictor_columns].loc[train_vector],(Z['metric2'].loc[train_vector])))
# print(rfr.score(Z[predictor_columns].loc[test_vector],(Z['metric2'].loc[test_vector])))
rfr.fit(Z_poly[train_vector],(Z['metric2_frac'].loc[train_vector]))
print(rfr.score(Z_poly[train_vector],(Z['metric2_frac'].loc[train_vector])))
print(rfr.score(poly.transform(Z[predictor_columns].loc[test_vector]),(Z['metric2_frac'].loc[test_vector])))
print()

model_dict = {
    'predictor_columns': predictor_columns,
    'model': rfr,
    'transform': poly
}

with open('value_regression.pkl', 'wb') as f:
    pickle.dump(model_dict,f)

Z['metric2_predict'] = rfr.predict(poly.transform(Z[predictor_columns]))

for i in range(12):
    o2 = []
    ss = 0
    # for i in range(len(tsvs)):
    tsv = o[test_inds[i]][0]
    col_no = o[test_inds[i]][1]
    A = Z.loc[(Z['DB']==tsv) & (Z['col_no']==col_no),['metric1','metric2','metric2_predict','index']]
    l = A.values.tolist()
    l.sort(key = lambda z: z[2], reverse=True)
    for ii in range(min(20,len(l))):
        print(l[ii])
    print()
    print()

print("\t".join([str(kk) for kk in l[0]]))
k = 0
if 1.5 in [i[0] for i in l]:
    best_ind = [i[0] for i in l].index(1.5)
    best_ind_score = l[best_ind][2]
else:
    best_ind = None
    best_ind_score = None
while (l[k][1]==1) and (k < len(l)):
    k += 1
if k == 0:
    o2.append((k,best_ind,best_ind_score,None,None,l[k][1],l[k][2]))
elif k == len(l):
    o2.append((k,best_ind,best_ind_score,l[k-1][1],l[k-1][2],None,None))
else:
    o2.append((k,best_ind,best_ind_score,l[k-1][1],l[k-1][2],l[k][1],l[k][2]))




