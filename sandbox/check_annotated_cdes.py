#!/usr/bin/python3

import os,pickle
import pandas as pd
from my_modules import utils, dataset_builders, score_functions, data_loader
import numpy as np


INPUT_DIR = "/home/cemarks/Projects/cancer/data/leaderboard"
DATABASE_ENDPOINT = "bolt://localhost:7688"
OUTPUT_PICKLE_FILE = "/home/cemarks/Projects/cancer/sandbox/expanded_training_X.pkl"


if __name__ == '__main__':
    g = utils.neo4j_connect(
        DATABASE_URI = DATABASE_ENDPOINT
    )
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
            annotated_result = score_functions.get_col_result_annotation(annotated,j+1)
            cde_id = score_functions.get_de_id(annotated_result)
            if cde_id is not None:
                q = "MATCH (n:CDE) WHERE n.CDE_ID = {0:d} RETURN ID(n)".format(cde_id)
                r = utils.query_graph(q,g)
                v = r.values()
                if len(v) == 0:
                    print(str(cde_id) + " " + str(f_tsv) + " " + str(j+1))



