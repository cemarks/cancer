#!/usr/bin/python3

import os
import pandas as pd

# Data ingest

def read_file(file_name, DATA_DIR = '/input'):
    files = os.listdir(DATA_DIR)
    if file_name in files:
        n,e = os.path.splitext(file_name)
        if e.lower() == '.tsv':
            df = pd.read_csv(os.path.join(DATA_DIR,file_name),sep="\t",header=0)
        elif e.lower() == '.csv':
            df = pd.read_csv(os.path.join(DATA_DIR,file_name),sep=",",header=0)
    elif file_name + ".tsv" in files:
        df = pd.read_csv(os.path.join(DATA_DIR,file_name + ".tsv"),sep="\t",header=0)
    elif file_name + ".csv" in files:
        df = pd.read_csv(os.path.join(DATA_DIR,file_name + ".csv"),sep=",",header=0)
    else:
        df = None
        raise FileNotFoundError("File Not Found or Unrecognized Format!")
    return df

def list_data_files(DATA_DIR = '/input'):
    files = os.listdir(DATA_DIR)
    data_files = [f for f in files if f[(len(f)-4):(len(f))].lower() in ['.csv','.tsv']]
    return data_files

def get_all_synonyms(graphDB):
    with graphDB.session() as q:
        string_tuples = q.run("MATCH (n:Synonym) RETURN n.name,n.lower")
    v = string_tuples.values()
    string_series = pd.DataFrame(data=v)
    string_series.columns = ['name','lower']
    return string_series
