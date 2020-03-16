import sys,os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase

if '/home/cemarks/Projects/cancer/docker-python/meta_challenge_py' not in sys.path:
    sys.path.append('/home/cemarks/Projects/cancer/docker-python/meta_challenge_py')

from data_loader import *

def full_str_match(col_name,g):
    str0 = [col_name.lower()]
    if col_name.replace("_"," ") != col_name:
        str0.append(col_name.replace("_"," "))
    if col_name.replace("."," ") != col_name:
        str0.append(col_name.replace("."," "))
    where_clause_list = ["n.lower = '" + i + "'" for i in str0]
    where_clause = " OR ".join(where_clause_list)
    with g.session() as q:
        z = q.run("MATCH (n:Synonym) WHERE " + where_clause + "RETURN n.name")
    return z.value()


def get_all_synonyms(g):
    with g.session() as q:
        string_tuples = q.run("MATCH (n:Synonym) RETURN n.name,n.lower")
    v = string_tuples.values()
    string_series = pd.DataFrame(data=v)
    string_series.columns = ['name','lower']
    return string_series


def find_words(col_name,synonyms_df):
    found_boolean = synonyms_df['lower'].apply(lambda z: z in col_name.lower())
    all_words = synonyms_df.loc[found_boolean]
    lengths = [(i,len(all_words['lower'].loc[i])) for i in all_words.index]
    lengths.sort(key=lambda z: z[1],reverse=True)
    i = 0
    while i < len(all_words):
        subs = ...





