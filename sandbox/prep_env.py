#!/usr/bin/python3

# Script to initiate python environment for command line development work & testin
# Need to operate in correct python virtualenv
# Neo4j database must be started and available on bolt://localhost/7687



import sys,os
import numpy as np
import pandas as pd
import json


sys.path.append(
    "/home/cemarks/Projects/cancer/docker-python/meta_challenge_py"
)

from my_modules import data_loader
from my_modules.utils import *
from my_modules.classifiers import *

graphdb = neo4j_connect(
    DATABASE_URI = "bolt://localhost:7688"
)

z = nameindex_query("gender",graphdb)


df = data_loader.read_file("caDSR-export-20190528-1320.tsv","/home/cemarks/Projects/cancer/data/Data/reference")
t = data_loader.read_file("Thesaurus.tsv","/home/cemarks/Projects/cancer/data/Data/reference")









q = "MATCH (n:AnswerText) - [] - (m:Answer) WHERE n.name_lower='yes' RETURN ID(n),ID(m)"
o = query_graph(q,graphdb)
oo = o.values()
len(oo) # 12755


q = "MATCH (n:AnswerText) - [] - (m:Answer) RETURN ID(n),ID(m)"
o = query_graph(q,G)
oo = o.values()
len(oo) # 12755


q = "MATCH (c:CDE) - [:PERMITS] - (n:Answer) - [eq:EQUALS] - (m:Concept) RETURN c.CDE_ID,eq.cde_id,c.name,n.name,m.CODE"
o = query_graph(q,G)
oo = o.values()
len(oo) # 12755




























