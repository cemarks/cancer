import pandas as pd
import os, pickle
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression,Lasso
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from sklearn import svm
from random import sample

from my_modules.learning_models import *

# WORKING_DIR = "/home/cemarks/Projects/cancer/sandbox"
# MODEL_DIR = "/home/cemarks/Projects/cancer/sandbox"

WORKING_DIR = "/home"
MODEL_DIR = WORKING_DIR
EPOCHS = 8
BATCH_SIZE = 2*4096

# Load data
with open(os.path.join(WORKING_DIR,"expanded_training_X.pkl"),'rb') as f:
    X = pickle.load(f)


# Get best value columns

# Only train on columns that have matches
X_VALUE_REGRESS = X.loc[X['metric2_max'] > 0]

X_VALUE_REGRESS['Rembrandt'] = (X_VALUE_REGRESS['DB'] == 'REMBRANDT-leaderboard').astype('int')
X_VALUE_REGRESS['Apollo'] = (X_VALUE_REGRESS['DB'] == 'APOLLO-2-leaderboard').astype('int')
X_VALUE_REGRESS['Outcome'] = (X_VALUE_REGRESS['DB'] == 'Outcome-Predictors-leaderboard').astype('int')
X_VALUE_REGRESS['ROI'] = (X_VALUE_REGRESS['DB'] == 'ROI-Masks-leaderboard').astype('int')

BEST_VALUE = 0.01

X_VALUE_REGRESS['Y'] = (X_VALUE_REGRESS['metric2']+X_VALUE_REGRESS['metric1']*BEST_VALUE/1.5)/(1+BEST_VALUE/1.5)
# X_VALUE_REGRESS['Y'] = (X_VALUE_REGRESS['metric2_frac'] > 0.8).astype('int')

# Separate into training & test
unique_columns=X_VALUE_REGRESS[['DB','col_no']].drop_duplicates().values
rand_ints = np.random.permutation(range(len(unique_columns)))

train_test_splitpoint = int(0.6*len(rand_ints))
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
        "secondary_search",
        "ftsearch_cde",
        "ftsearch_dec",
        # "syn_classsum",
        "syn_propsum",
        # "syn_objsum",
        # "syn_classmax",
        # "syn_propmax",
        "syn_objmax",
        "ftsearch_question",
        "enum_concept_search",
        "enum_answer_search",
        # "answer_count_score",
        "value_score",
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
        "pct_secondary_search",
        "cde_frac",
        "dec_frac",
        "que_frac",
        # "syn_classsum_frac",
        "syn_propsum_frac",
        # "syn_objsum_frac",
        # "syn_classmax_frac",
        # "syn_propmax_frac",
        "syn_objmax_frac",
        "enum_concept_frac",
        "enum_ans_frac",
        # "ans_score_frac",
        "val_score_frac",
        "cde_ecdf",
        "dec_ecdf",
        # "syn_classsum_ecdf",
        "syn_propsum_ecdf",
        # "syn_objsum_ecdf",
        # "syn_classmax_ecdf",
        # "syn_propmax_ecdf",
        "syn_objmax_ecdf",
        "question_ecdf",
        "enum_concept_ecdf",
        "enum_ans_ecdf",
        # "ans_score_ecdf",
        "val_score_ecdf",
        "n",
        # "logn"
    ]
    # poly = PolynomialFeatures(degree = 2)
    # Z_poly = poly.fit_transform(x[predictor_columns])
    Z_poly = x[predictor_columns]
    return Z_poly



XX = rr_transform(X_VALUE_REGRESS.loc[train_vector])
XT = rr_transform(X_VALUE_REGRESS.loc[test_vector])


from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



scaler = MinMaxScaler()

# XX = scaler.fit_transform(XX)
# XT = scaler.transform(XT)
XX = scaler.fit_transform(XX)
XT = scaler.transform(XT)

BASE = 2
TANSHIFT = 5/6
EXP = 100


# YY = (EXP ** X_VALUE_REGRESS['Y'].loc[train_vector] - 1)/(EXP-1)
# YT = (EXP ** X_VALUE_REGRESS['Y'].loc[test_vector] - 1)/(EXP-1)
# YY = np.log((BASE-1)*X_VALUE_REGRESS['metric2_frac'].loc[train_vector]+1)/np.log(BASE)
# YT = np.log((BASE-1)*X_VALUE_REGRESS['metric2_frac'].loc[test_vector]+1)/np.log(BASE)
# YY = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['metric2_frac'].loc[train_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YT = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['metric2_frac'].loc[test_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YY = X_VALUE_REGRESS['metric2_frac'].loc[train_vector]
# YT = X_VALUE_REGRESS['metric2_frac'].loc[test_vector]
YY = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['Y'].loc[train_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
YT = np.tan((TANSHIFT * np.pi)*X_VALUE_REGRESS['Y'].loc[test_vector] - (TANSHIFT)*(np.pi/2))/np.tan((TANSHIFT)*(np.pi/2))
# YY = X_VALUE_REGRESS['Y'].loc[train_vector]
# YT = X_VALUE_REGRESS['Y'].loc[test_vector]
# YY = np.log((BASE-1)*X_VALUE_REGRESS['Y'].loc[train_vector]+1)/np.log(BASE)
# YT = np.log((BASE-1)*X_VALUE_REGRESS['Y'].loc[test_vector]+1)/np.log(BASE)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(XX.shape[1],)),
    # tf.keras.layers.Dense(256, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
    # tf.keras.layers.Dropout(0),
    tf.keras.layers.Dense(48, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(l=0)),
    tf.keras.layers.Dropout(0),
    tf.keras.layers.Dense(12, activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(l=0)),
    tf.keras.layers.Dropout(0),
    # tf.keras.layers.Dense(8, activation = 'relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='tanh')
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(0.0004),
    loss = 'mse',
    metrics = ['mae','mse']
)

# model.compile(
#     optimizer=tf.keras.optimizers.RMSprop(0.0004),
#     loss = 'binary_crossentropy',
#     metrics = ['accuracy']
# )

history = model.fit(
    x=XX,
    y=YY,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(XT,YT),
)


# Save to model dir
model_dict = {
    'scalertransformer': scaler,
    'var_count': XX.shape[1]
}


with open(os.path.join(MODEL_DIR,'value_nn_scaler.pkl'), 'wb') as f:
    pickle.dump(model_dict,f)

model.save_weights(os.path.join(MODEL_DIR,"nn-weights.h5"),save_format="h5")


# Print ordering of test set results.
# X_VALUE_REGRESS['metric2_predict'] = model.predict(
#     scaler.transform(
#         rr_transform(X_VALUE_REGRESS)
#     )
# )   

NEWX = X_VALUE_REGRESS.loc[test_vector]
NEWX['metric2_predict'] = model.predict(
    scaler.transform(
        rr_transform(NEWX)
    )
)   

metrics = []
with open(os.path.join(MODEL_DIR,"output.txt"),'w') as f:
    for i in range(len(test_inds)):
        tsv = unique_columns[test_inds[i]][0]
        col_no = unique_columns[test_inds[i]][1]
        display_df = NEWX.loc[(NEWX['DB']==tsv) & (NEWX['col_no']==col_no),['metric1','metric2','Y','metric2_predict','index']]
        display_list = display_df.values.tolist()
        display_list.sort(key = lambda z: z[3], reverse=True)
        metrics.append(
            np.max(
                [
                    (1 + (2-i)/1.5)*display_list[0][0] + (3-i)*display_list[0][1] for i in range(3)
                ]
            )

        )
        for ii in range(min(20,len(display_list))):
            f.write(str(display_list[ii])+"\n")
        f.write("\n\n")

print("SCORE: {0:1.3f}".format(np.mean(metrics)))

