from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge as ridge_regression
from sklearn.linear_model import LogisticRegression as logistic_regression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import os, pickle
import numpy as np
from random import sample
from sklearn.model_selection import train_test_split

os.chdir("/home/cemarks/Projects/cancer/sandbox")

with open("expanded_X.pkl",'rb') as f:
    X = pickle.load(f)


# X['max_cde'] = 0
# X['max_dec'] = 0
# X['max_que'] = 0
# X['max_syn_class'] = 0
# X['max_syn_prop'] = 0
# X['max_syn_obj'] = 0
# X['max_enum_concept'] = 0
# X['max_enum_ans'] = 0
# X['n'] = 0
# X['pct_cde'] = 0
# X['pct_dec'] = 0
# X['pct_que'] = 0
# X['pct_syn_class'] = 0
# X['pct_syn_prop'] = 0
# X['pct_syn_obj'] = 0
# X['pct_enum_concept'] = 0
# X['pct_enum_ans'] = 0
# X['pct_ans_score'] = 0
# X['pct_val_score'] = 0
# X['cde_frac'] = 0
# X['dec_frac'] = 0
# X['que_frac'] = 0
# X['syn_prop_frac'] = 0
# X['syn_obj_frac'] = 0
# X['enum_concept_frac'] = 0
# X['enum_ans_frac'] = 0
# X['ans_score_frac'] = 0
# X['val_score_frac'] = 0
# X['metric2_max'] = 0
# X['metric2_frac'] = 0
# X['pct_metric2_pos'] = 0


# for i in X['DB'].unique():
#     for j in X['col_no'].loc[X['DB'] == i].unique():
#         inds = (X['DB'] == i) & (X['col_no'] == j)
#         max_cde = max(X['ftsearch_cde'].loc[inds])
#         max_dec = max(X['ftsearch_dec'].loc[inds])
#         max_que = max(X['ftsearch_question'].loc[inds])
#         max_syn_class = max(X['ftsearch_syn_class'].loc[inds])
#         max_syn_prop = max(X['ftsearch_syn_prop'].loc[inds])
#         max_syn_obj = max(X['ftsearch_syn_obj'].loc[inds])
#         max_enum_concept = max(X['enum_concept_search'].loc[inds])
#         max_enum_ans = max(X['enum_answer_search'].loc[inds])
#         max_ans_score = max(X['answer_count_score'].loc[inds])
#         max_val_score = max(X['value_score'].loc[inds])
#         max_metric2 = max(X['metric2'].loc[inds])
#         n = sum(inds)
#         X.loc[inds,'max_cde'] = max_cde
#         X.loc[inds,'max_dec'] = max_dec
#         X.loc[inds,'max_que'] = max_que
#         X.loc[inds,'max_syn_class'] = max_syn_class
#         X.loc[inds,'max_syn_prop'] = max_syn_prop
#         X.loc[inds,'max_syn_obj'] = max_syn_obj
#         X.loc[inds,'max_enum_concept'] = max_enum_concept
#         X.loc[inds,'max_enum_ans'] = max_enum_ans
#         X.loc[inds,'max_ans_score'] = max_ans_score
#         X.loc[inds,'max_val_score'] = max_val_score
#         X.loc[inds,'metric2_max'] = max_metric2
#         X.loc[inds,'n'] = n
#         X.loc[inds,'pct_cde'] = sum(inds & X['ftsearch_cde'] > 0)/n
#         X.loc[inds,'pct_dec'] = sum(inds & X['ftsearch_dec'] > 0)/n
#         X.loc[inds,'pct_que'] = sum(inds & X['ftsearch_question'] > 0)/n
#         X.loc[inds,'pct_syn_class'] = sum(inds & X['ftsearch_syn_class'] > 0)/n
#         X.loc[inds,'pct_syn_prop'] = sum(inds & X['ftsearch_syn_prop'] > 0)/n
#         X.loc[inds,'pct_syn_obj'] = sum(inds & X['ftsearch_syn_obj'] > 0)/n
#         X.loc[inds,'pct_enum_concept'] = sum(inds & X['enum_concept_search'] > 0)/n
#         X.loc[inds,'pct_enum_ans'] = sum(inds & X['enum_answer_search'] > 0)/n
#         X.loc[inds,'pct_ans_score'] = sum(inds & X['answer_count_score'] > 0)/n
#         X.loc[inds,'pct_val_score'] = sum(inds & X['value_score'] > 0)/n
#         X.loc[inds,'pct_metric2_pos'] = sum(inds & X['metric2'] > 0)/n
#         X.loc[inds,'cde_frac'] = 0 if max_cde == 0 else X.loc[inds,'ftsearch_cde']/max_cde
#         X.loc[inds,'dec_frac'] = 0 if max_dec == 0 else X.loc[inds,'ftsearch_dec']/max_dec
#         X.loc[inds,'que_frac'] = 0 if max_que == 0 else X.loc[inds,'ftsearch_question']/max_que
#         X.loc[inds,'syn_class_frac'] = 0 if max_syn_class == 0 else X.loc[inds,'ftsearch_syn_class']/max_syn_class
#         X.loc[inds,'syn_prop_frac'] = 0 if max_syn_prop == 0 else X.loc[inds,'ftsearch_syn_prop']/max_syn_prop
#         X.loc[inds,'syn_obj_frac'] = 0 if max_syn_obj == 0 else X.loc[inds,'ftsearch_syn_obj']/max_syn_obj
#         X.loc[inds,'enum_concept_frac'] = 0 if max_enum_concept == 0 else X.loc[inds,'enum_concept_search']/max_enum_concept
#         X.loc[inds,'enum_ans_frac'] = 0 if max_enum_ans == 0 else X.loc[inds,'enum_answer_search']/max_enum_ans
#         X.loc[inds,'ans_score_frac'] = 0 if max_ans_score == 0 else X.loc[inds,'answer_count_score']/max_ans_score
#         X.loc[inds,'val_score_frac'] = 0 if max_val_score == 0 else X.loc[inds,'value_score']/max_val_score
#         X.loc[inds,'metric2_frac'] = 0 if max_metric2 == 0 else X.loc[inds,'metric2']/max_metric2


# print("\n".join([str(i) + " " + X.columns[i] for i in range(X.shape[1])]))
# # X['log_cde'] = X.apply(lambda z: z[7]/np.log(1+z[16]),axis=1)

X['metric3'] = (X['metric1'] > 1).astype('int')
X['metric4'] = (X['metric2'] > 0).astype('int')
X['metric5'] = (X['metric2_frac'] > 0.5).astype('int')

not_founds = [
    (6428091, 0, 9),
    (6154731, 1, 32),
    (6002302, 1, 37),
    (6154728, 1, 40),
    (6154731, 2, 29),
    (2608243, 2, 30),
    (5143957, 2, 32)
]

INPUT_DIR = "/home/cemarks/Projects/cancer/data/leaderboard"
file_names = os.listdir(INPUT_DIR)
file_splits = [os.path.splitext(f) for f in file_names]
tsvs = [i for i in file_splits if i[1] == '.tsv']
dbs = [i[0] for i in tsvs]
found_inds = (X['DB'] != tsvs[not_founds[0][1]][0]) | (X['col_no'] != not_founds[0][2])
for j in not_founds[1:len(not_founds)]:
    found_inds = found_inds & ((X['DB'] != tsvs[j[1]][0]) | (X['col_no'] != j[2]))


Y = X.loc[found_inds]



predictor_columns = [
#    "ftsearch_syn_class",
    # "ftsearch_syn_prop",
    # "ftsearch_syn_obj",
    # "ftsearch_cde",
    # "ftsearch_dec",
    # "ftsearch_question",
    # "enum_concept_search",
    # "enum_answer_search",
    # "answer_count_score",
    # "value_score",
    "max_cde",
    "max_dec",
    "max_que",
    "max_syn_class",
    "max_syn_prop",
    "max_syn_obj",
    "max_enum_concept",
    "max_enum_ans",
    "max_ans_score",
    "max_val_score",
    "pct_cde",
    "pct_dec",
    "pct_que",
    "pct_syn_class",
    "pct_syn_prop",
    "pct_syn_obj",
    "pct_enum_concept",
    "pct_enum_ans",
    "pct_ans_score",
    "pct_val_score",
    "cde_frac",
    "dec_frac",
    "que_frac",
    "syn_class_frac",
    "syn_prop_frac",
    "syn_obj_frac",
    "enum_concept_frac",
    "enum_ans_frac",
    "ans_score_frac",
    "val_score_frac",
    "n"

]

# predictor_columns = X.columns[4:14]
lin = ridge_regression(alpha=1)
lin.fit(X[predictor_columns],X['metric2'])
s = lin.score(X[predictor_columns],X['metric2'])


poly = PolynomialFeatures(degree = 2)
Y_poly = poly.fit_transform(Y[predictor_columns])
lin = ridge_regression(alpha=100)
lin.fit(Y_poly,Y['metric2_frac'])
s = lin.score(Y_poly,Y['metric2_frac'])
print(s)

logreg = logistic_regression(
    class_weight = {
        0:0.04474471247694264,
        1:0.9552552875230573
    },
    C = 0.1
)
logreg.fit(X_poly,X['metric4'])
s = logreg.score(X_poly,X['metric4'])
print(s)




poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(Y[predictor_columns])
# lin = ridge_regression(alpha=10)
# lin.fit(X_poly,X['metric2'])
# s = lin.score(X_poly,X['metric2'])
# print(s)

logreg = logistic_regression(
    class_weight = {
        0:0.0002749811274745059,
        1:0.9997250188725255
    },
    C = 0.1
)
logreg.fit(X_poly,Y['metric5'])
s = logreg.score(X_poly,Y['metric5'])
print(s)



logreg = logistic_regression(
    class_weight = {
        0:0.04760675728928941,
        1:0.9523932427107106
    },
    C = 0.1
)
logreg.fit(X_poly,Y['metric4'])
s = logreg.score(X_poly,Y['metric4'])
print(s)




z = []
for i in X['DB'].unique():
    for j in X['col_no'].loc[X['DB'] == i].unique():
        inds = (X['DB'] == i) & (X['col_no'] == j)
        z.append(max(X['metric2'].loc[inds]))




z = []
for i in X['DB'].unique():
    for j in X['col_no'].loc[X['DB'] == i].unique():
        z.append((i,j))


z = []
stat_columns = for i in Y['DB'].unique():
    for j in Y['col_no'].loc[Y['DB'] == i].unique():
        z.append(Y.loc[(Y['DB']==i) & (Y['col_no'] == j),stat_columns].iloc[0])



predictor_columns = [
    "max_cde",
    "max_dec",
    "max_que",
    "max_syn_class",
    "max_syn_prop",
    "max_syn_obj",
    "max_enum_concept",
    "max_enum_ans",
    "max_ans_score",
    "max_val_score",
    # "pct_cde",
    # "pct_dec",
    # "pct_que",
    # "pct_syn_class",
    # "pct_syn_prop",
    # "pct_syn_obj",
    # "pct_enum_concept",
    # "pct_enum_ans",
    # "pct_ans_score",
    # "pct_val_score",
    "n"
]

D = pd.DataFrame(z)
D['Y'] = D.apply(lambda z: 1 if z[21] == 0 else 0, axis=1) #These are nomatch rows.


log_transforms = [
    # "max_cde",
    # "max_dec",
    # "max_que",
    # "max_syn_class",
    # "max_syn_prop",
    # "max_syn_obj",
    # "max_enum_concept",
    # "max_enum_ans",
    # "max_ans_score",
    # "max_val_score",
    "n"
]

for l in log_transforms:
    D[l] = np.log(D[l]+1)

scaler = preprocessing.MinMaxScaler()
scaler.fit(D[predictor_columns])

poly = PolynomialFeatures(degree = 2)

test_inds = sample(range(D.shape[0]),int(0.1*D.shape[0]))
tng_inds = [i for i in range(D.shape[0]) if i not in test_inds]

# XX = scaler.transform(D[predictor_columns])
XX = D[predictor_columns].iloc[tng_inds]
XT = D[predictor_columns].iloc[test_inds]

YY = D['Y'].iloc[tng_inds]
YT = D['Y'].iloc[test_inds]

X_poly = poly.fit_transform(XX)
# logreg = logistic_regression(
#     C = 1,
#     max_iter = 1000,
#     tol=0.000000001,
#     solver='liblinear'
# )
# logreg.fit(D[predictor_columns],D['Y'])
# s = logreg.score(D[predictor_columns],D['Y'])
# print(s)

# logreg.fit(X_poly,D['Y'])
# s = logreg.score(X_poly,D['Y'])
# print(s)


# from sklearn.model_selection import KFold
# from sklearn import preprocessing


kf = KFold()

INDS = kf.split(X_poly)
o = []
ot = []
for train_index,test_index in INDS:
    s = []
    st = []
    # for c in [0.00002,0.0001,0.0005,0.001,0.002]:
    for c in [0.001]:
        logreg = logistic_regression(
            C = c,
            max_iter = 1000,
            tol=0.0001,
            solver='newton-cg'
        )
        logreg.fit(X_poly[train_index],YY.iloc[train_index])
        s.append(logreg.score(X_poly[test_index],YY.iloc[test_index]))
        st.append(logreg.score(X_poly[train_index],YY.iloc[train_index]))
    o.append(s)
    ot.append(st)

o = np.array(o)
m = np.mean(o, axis=0)
print(m)
ot = np.array(ot)
mt = np.mean(ot, axis=0)
print(mt)


best_model = logistic_regression(
    C = 0.001,
    max_iter = 1000,
    tol=0.0001,
    solver='newton-cg'
)

best_model.fit(X_poly,YY)
print(best_model.score(poly.transform(XT),YT))

a = ensemble.AdaBoostClassifier(best_model)
a.fit(X_poly,YY)
print(a.score(poly.transform(XT),YT))

rf = RandomForestClassifier(
    n_estimators=10,
    criterion='entropy',
    max_features = 'auto'
)
rf.fit(XX,YY)
print(rf.score(XT,YT))
rf.fit(X_poly,YY)
print(rf.score(poly.transform(XT),YT))

from matplotlib import pyplot as plt

probs = best_model.predict_proba(
    poly.transform(XT)
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



#################3

# Round II: for those with matches, get best for round 3

Z = Y.loc[Y['metric2_max'] > 0]


predictor_columns = [
    # "ftsearch_syn_class",
    "ftsearch_syn_prop",
    # "ftsearch_syn_obj",
    "ftsearch_cde",
    "ftsearch_dec",
    "ftsearch_question",
    "enum_concept_search",
    "enum_answer_search",
    "answer_count_score",
    "value_score",
    # "max_cde",
    # "max_dec",
    # "max_que",
    # "max_syn_class",
    # "max_syn_prop",
    # "max_syn_obj",
    # "max_enum_concept",
    # "max_enum_ans",
    # "max_ans_score",
    # "max_val_score",
    # "pct_cde",
    # "pct_dec",
    # "pct_que",
    # "pct_syn_class",
    # "pct_syn_prop",
    # "pct_syn_obj",
    # "pct_enum_concept",
    # "pct_enum_ans",
    # "pct_ans_score",
    # "pct_val_score",
    "cde_frac",
    "dec_frac",
    "que_frac",
    "syn_class_frac",
    "syn_prop_frac",
    "syn_obj_frac",
    "enum_concept_frac",
    "enum_ans_frac",
    "ans_score_frac",
    "val_score_frac",
    "n",
    # "logn"
]

Z['logn'] = np.log(Z['n'])

poly = PolynomialFeatures(degree = 2)

cutoff = 0.5
Z['metric5'] = (Z['metric2_frac'] > cutoff).astype('int')
Z_poly = poly.fit_transform(Z[predictor_columns])

n_pos = sum(Z['metric5'])
wt = n_pos/len(Z)

# logreg = logistic_regression(
#     class_weight = {
#         0:wt,
#         1:1-wt
#     },
#     C = 1000,
#     tol = 0.00001,
#     max_iter = 1000,
#     solver = 'newton-cg'
# )
# logreg.fit(Z_poly,Z['metric5'])
# s = logreg.score(Z_poly,Z['metric5'])
# print(s)

rf = RandomForestClassifier(
    n_estimators = 50,
    class_weight={
        0: wt,
        1: 1-wt
    }
)
rf.fit(Z_poly,Z['metric5'])
print(rf.score(Z_poly,Z['metric5']))

rf = RandomForestRegressor(
    n_estimators = 50
)
rf.fit(Z_poly,Z['metric2'])
print(rf.score(Z_poly,Z['metric2']))


Z['metric4'] = Z.apply(lambda z: 0 if z[15] == 0 else 1, axis=1)
n_pos = sum(Z['metric4'])

print(n_pos)

wt = n_pos/len(Z)


# pos_inds = [i for i, x in enumerate((Z['metric4'] == 1).tolist()) if x]
# neg_inds = [i for i, x in enumerate((Z['metric4'] == 0).tolist()) if x]

# pos_test = sample(pos_inds, int(0.15*len(pos_inds)))
# neg_test = sample(neg_inds, int(0.15*len(neg_inds)))

# training_inds = list(set(Z.index.tolist()) - set(pos_test+neg_test))
# test_inds = pos_test + neg_test

rf = RandomForestClassifier(
    n_estimators = 100,
    class_weight={
        0: wt,
        1: 1-wt
    }
)
rf.fit(Z[predictor_columns].iloc[training_inds],Z['metric4'].iloc[training_inds])
print(rf.score(Z[predictor_columns].iloc[training_inds],Z['metric4'].iloc[training_inds]))
print(rf.score(Z[predictor_columns].iloc[test_inds],Z['metric4'].iloc[test_inds]))
print(rf.score(Z[predictor_columns].iloc[pos_test],Z['metric4'].iloc[pos_test]))


rf.fit(Z_poly[training_inds],Z['metric4'].iloc[training_inds])
print(rf.score(Z_poly[training_inds],Z['metric4'].iloc[training_inds]))
print(rf.score(Z_poly[test_inds],Z['metric4'].iloc[test_inds]))
print(rf.score(Z_poly[pos_inds],Z['metric4'].iloc[pos_inds]))



rand_ints = np.random.permutation(range(len(Z)))
train_test_splitpoint = int(0.85*len(Z))
training_inds = rand_ints[0:train_test_splitpoint]
test_inds = rand_ints[train_test_splitpoint:len(rand_ints)]

rfr = RandomForestRegressor(
    n_estimators = 50
)
rfr.fit(Z[predictor_columns].iloc[training_inds],(Z['metric2'].iloc[training_inds]))
print(rfr.score(Z[predictor_columns].iloc[training_inds],(Z['metric2'].iloc[training_inds])))
print(rfr.score(Z[predictor_columns].iloc[test_inds],(Z['metric2'].iloc[test_inds])))

# rfr.fit(Z_poly[training_inds],Z['metric2'].iloc[training_inds])
# print(rfr.score(Z_poly[training_inds],Z['metric2'].iloc[training_inds]))
# print(rfr.score(Z_poly[test_inds],Z['metric2'].iloc[test_inds]))


rdr = ridge_regression(
    100
)
rdr.fit(Z[predictor_columns].iloc[training_inds],(Z['metric2'].iloc[training_inds]))
print(rdr.score(Z[predictor_columns].iloc[training_inds],(Z['metric2'].iloc[training_inds])))
print(rdr.score(Z[predictor_columns].iloc[test_inds],(Z['metric2'].iloc[test_inds])))

# rdr.fit(Z_poly[training_inds],Z['metric2'].iloc[training_inds])
# print(rdr.score(Z_poly[training_inds],Z['metric2'].iloc[training_inds]))
# print(rdr.score(Z_poly[test_inds],Z['metric2'].iloc[test_inds]))





Z['metric2_predict'] = rfr.predict(Z[predictor_columns])

i = 0
tsv = tsvs[i][0]
J = Z['col_no'].loc[Z['DB']==tsv].unique()
j = 11
col_no = J[j]
A = Z.loc[(Z['DB']==tsv) & (Z['col_no']==col_no),['metric2','metric2_predict']]
l = A.values.tolist()
l.sort(key = lambda z: z[1], reverse=True)

