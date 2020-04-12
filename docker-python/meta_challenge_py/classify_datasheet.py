#!/usr/bin/python3

import sys,os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
import json,pickle
from my_modules import data_loader,utils,dataset_builders,result_builders

# DATABASE_URI = "bolt://localhost:7687"
# DATABASE_USER = "neo4j"
# DATABASE_PASSWORD = "loHmjZWp"
# INPUT_FOLDER = "/input"
# OUTPUT_FOLDER = "/output"
# NOMATCH_MODEL_PATH = '/home/cemarks/Projects/cancer/sandbox/nomatch_mod.pkl'
# VALUE_MODEL_PATH = '/home/cemarks/Projects/cancer/sandbox/value_regression.pkl'

DATABASE_URI = "bolt://localhost:7688"
DATABASE_USER = "neo4j"
DATABASE_PASSWORD = "loHmjZWp"
INPUT_FOLDER = "/home/cemarks/Projects/cancer/mount_folder/input"
OUTPUT_FOLDER = "/home/cemarks/Projects/cancer/mount_folder/output"
NOMATCH_MODEL_PATH = '/home/cemarks/Projects/cancer/sandbox/nomatch_model.pkl'
VALUE_MODEL_PATH = '/home/cemarks/Projects/cancer/sandbox/value_regression.pkl'

# Nomatch model output cutoffs
NOMATCH_CUTOFFS = [0.4,0.67,1]

def nomatch_prob(column_df,nomatch_model):
    predictor_columns = nomatch_model['predictor_columns']
    if 'transform' in nomatch_model:
        X = nomatch_model['transform'].transform(column_df[predictor_columns].iloc[0])
    else:
        X = column_df[predictor_columns].iloc[0]
    X_reshape = X.values.reshape(1,-1)
    predictor_model = nomatch_model['model']
    p = predictor_model.predict_proba(X_reshape)
    return(p[0][1])

def estimate_value(column_df,value_model):
    predictor_columns = value_model['predictor_columns']
    if 'transform' in value_model:
        X = value_model['transform'].transform(column_df[predictor_columns])
    else:
        X = column_df[predictor_columns].values
    predictor_model = value_model['model']
    Y = predictor_model.predict(X)
    return(Y)


def classify_column(column_df,nomatch_model,value_model):
    if column_df.shape[0] > 0:
        nm_prob = nomatch_prob(column_df,nomatch_model)
        value_ests = estimate_value(column_df,value_model)
        appended_df = column_df.copy()
        appended_df['value_est'] = value_ests
        value_est_index = appended_df.columns.tolist().index('value_est')
        appended_df.sort_values(by='value_est',axis=0, ascending=False, inplace=True)
        top3 = [int(appended_df['cde_id'].iloc[0])]
        i = 1
        while (len(top3) < 3) and (i < appended_df.shape[0]):
            new_top = int(appended_df['cde_id'].iloc[i])
            if new_top not in top3:
                top3.append(new_top)
            i += 1
        if nm_prob > NOMATCH_CUTOFFS[0]:
            classification = ['nomatch'] + top3[0:2]
        elif nm_prob > NOMATCH_CUTOFFS[1]:
            classification = [top3[0]] + ['nomatch'] + top3[1:2]
        elif nm_prob > NOMATCH_CUTOFFS[2]:
            classification = top3[0:2] + ['nomatch']
        else: classification = top3
    else:
        classification = ['nomatch']
    return classification

def classify_columns(data_sheet,g,nomatch_model,value_model):
    columns = []
    for i,c in enumerate(data_sheet.columns.tolist()):
        col_series = data_sheet[c]
        unique_values = utils.col_unique_values(col_series)
        column_df = dataset_builders.build_expanded_X(
            col_series,
            g
        )
        class_list = classify_column(
            column_df,
            nomatch_model,
            value_model
        )
        results = result_builders.create_result_array(
            class_list,
            unique_values,
            g
        )
        d = {
            "columnNumber": i+1,
            "headerValue": c,
            "results": results
        }
        columns.append(d)
    columns_obj = {
        "columns": columns
    }
    return columns_obj


if __name__=="__main__":
    graphDB = utils.neo4j_connect(
        DATABASE_URI,
        DATABASE_USER,
        DATABASE_PASSWORD
    )
    with open(NOMATCH_MODEL_PATH,'rb') as f:
        nomatch_model = pickle.load(f)
    with open(VALUE_MODEL_PATH,'rb') as f:
        value_model = pickle.load(f)
    if len(sys.argv) > 1:
        data_files = [sys.argv[1]]
    else:
        data_files = data_loader.list_data_files(DATA_DIR = INPUT_FOLDER)
    for data_file in data_files:
        file_split = os.path.splitext(data_file)
        file_base = file_split[0]
        output_file_name = os.path.join(OUTPUT_FOLDER,"{0:s}-Submission.json".format(file_base))
        df = data_loader.read_file(data_file,INPUT_FOLDER)
        columns_obj = classify_columns(df,graphDB,nomatch_model,value_model)
        with open(output_file_name,'w') as f:
            json.dump(columns_obj,f)



