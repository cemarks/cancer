import sys,os
import numpy as np
import pandas as pd
import json
import pickle

from my_modules import data_loader
from my_modules.utils import *
from my_modules.classifiers import *
from my_modules import score_functions

g = neo4j_connect(
    DATABASE_URI = "bolt://localhost:7688"
)


INPUT_DIR = "/home/cemarks/Projects/cancer/data/leaderboard"
file_names = os.listdir(INPUT_DIR)
file_splits = [os.path.splitext(f) for f in file_names]
tsvs = [i for i in file_splits if i[1] == '.tsv']


X = pd.DataFrame()
for i in range(len(tsvs)):
    f_tsv = "".join(tsvs[i])
    data_sheet = data_loader.read_file(f_tsv,INPUT_DIR)
    a_json_index = file_splits.index(("Annotated-{0:s}".format(tsvs[i][0]),".json"))
    f_annotated = "".join(file_splits[a_json_index])
    annotated = score_functions.read_annotation(os.path.join(INPUT_DIR,f_annotated))
    for j in range(len(data_sheet.columns)):
        col_series = data_sheet[data_sheet.columns[j]]
        annotated_result = score_functions.get_col_result_annotation(annotated,j+1)
        D = build_X(col_series,annotated_result,g)
        D.insert(0,'DB',[tsvs[i][0]] * D.shape[0])
        D.insert(1,'col_no',[j+1] * D.shape[0])
        X = pd.concat([X,D])


with open("/home/cemarks/Projects/cancer/sandbox/ml_dataframe.pkl","wb") as f_save:
    pickle.dump(X,f_save)


