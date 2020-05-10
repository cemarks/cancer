#!/usr/bin/python3

import os
import json
import pickle
import numpy as np
from my_modules import learning_models,score_functions


WORKING_DIR = "/home/cemarks/Projects/cancer/sandbox"
MODEL_DIR = "/home/cemarks/Projects/cancer/sandbox"
ANNOTATION_JSON_DIR = "/home/cemarks/Projects/cancer/data/leaderboard"

NOMATCH_CUTOFFS = [0.19,0.23,0.48]

def nomatch_score(de_name,result_no):
    if de_name == "NOMATCH":
        return 3 + (2 - result_no)
    else:
        return 0

def match_score(db_row,result_no):
    score = db_row['metric1'] + db_row['metric2'] + (db_row['value_score'] * 0.5)
    bonus = (2 - result_no) * np.mean([db_row['metric1'],db_row['metric2'],db_row['value_score']])
    return score + bonus

# Load data
with open(os.path.join(WORKING_DIR,"expanded_training_X.pkl"),'rb') as f:
    X = pickle.load(f)

with open(os.path.join(MODEL_DIR,"nomatch_model.pkl"),'rb') as f:
    nomatch_model = pickle.load(f)

with open(os.path.join(MODEL_DIR,"value_regression.pkl"),'rb') as f:
    value_model = pickle.load(f)

file_scores = []
for score_file in list(X['DB'].unique()):
    col_scores = []
    with open(os.path.join(ANNOTATION_JSON_DIR,"Annotated-{0:s}.json".format(score_file)),'r') as f:
        annotated_json = json.load(f)
    for col_no in X['col_no'].loc[X['DB']==score_file].unique():
        ############
        column_df = X.loc[(X['DB'] == score_file) & (X['col_no'] == col_no)]
        col_annotation = score_functions.get_col_result_annotation(annotated_json,col_no)
        de_name = col_annotation['dataElement']['name']
        if column_df.shape[0] > 0:
            nomatch_predictor = nomatch_model['model']
            nomatch_probs = nomatch_predictor.predict_proba(learning_models.lr_transform(column_df))
            nm_prob = nomatch_probs[0][1]
            value_predictor = value_model['model']
            value_ests = value_predictor.predict(learning_models.rr_transform(column_df))
            column_df.insert(
                column_df.shape[1],
                'value_est',
                value_ests
            )
            column_df = column_df.sort_values(by='value_est',axis=0,ascending=False)
            if nm_prob < NOMATCH_CUTOFFS[0]:
                scores = [
                    match_score(column_df.iloc[0],1),
                    match_score(column_df.iloc[1],2),
                    match_score(column_df.iloc[2],3)
                ]
            elif nm_prob < NOMATCH_CUTOFFS[1]:
                scores = [
                    match_score(column_df.iloc[0],1),
                    match_score(column_df.iloc[1],2),
                    nomatch_score(de_name,3)
                ]
            elif nm_prob < NOMATCH_CUTOFFS[2]:
                scores = [
                    match_score(column_df.iloc[0],1),
                    nomatch_score(de_name,2),
                    match_score(column_df.iloc[1],3)
                ]
            else: 
                scores = [
                    nomatch_score(de_name,1),
                    match_score(column_df.iloc[0],2),
                    match_score(column_df.iloc[1],3)
                ]
        else:
            scores = [
                nomatch_score(de_name,1)
            ]
        col_score = np.max(scores)
        col_scores.append(col_score)
        #####################
    file_score = np.mean(col_scores)
    file_scores.append((score_file,file_score))


for fs in file_scores:
    print("{0:s}: {1:1.5f}".format(fs[0],fs[1]))


