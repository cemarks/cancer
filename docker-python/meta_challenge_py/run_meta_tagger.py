#!/usr/bin/python3

import sys,os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
import json

from my_modules import data_loader
from my_modules.utils import *


DATABASE_URI = "bolt://localhost:7687"
DATABASE_USER = "neo4j"
DATABASE_PASSWORD = "loHmjZWp"
INPUT_FOLDER = "/input"


if __name__=="__main__":
    graphDB = neo4j_connect(
        DATABASE_URI,
        DATABASE_USER,
        DATABASE_PASSWORD
    )
    synonyms = data_loader.get_all_synonyms(graphDB)
    if len sys.argv > 1:
        data_files = [sys.argv[1]]
    else:
        data_files = data_loader.list_data_files()
    for data_file in data_files:
        df = data_loader.read_file(data_file,INPUT_FOLDER)
        columns_obj = classifiers.classify_columns(df,graphDB)
        values_obj = classifiers.classify_values(df,columns_obj,graphDB)
        output_json = json_writer.format_output(columns_obj,values_obj)
        json_writer.write_json_output(data_file,output_json)

