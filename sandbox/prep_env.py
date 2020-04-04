#!/usr/bin/python3

# Script to initiate python environment for command line development work & testin
# Need to operate in correct python virtualenv
# Neo4j database must be started and available on bolt://localhost/7687



import sys,os
import numpy as np
import pandas as pd
import json

from my_modules import data_loader
from my_modules.utils import *
from my_modules.classifiers import *
from my_modules import score_functions

graphdb = neo4j_connect(
    DATABASE_URI = "bolt://localhost:7688"
)

z = nameindex_query("gender",graphdb)


cadsr_df = data_loader.read_file("caDSR-export-20190528-1320.tsv","/home/cemarks/Projects/cancer/data/Data/reference")
t = data_loader.read_file("Thesaurus.tsv","/home/cemarks/Projects/cancer/data/Data/reference")


data_sheet = data_loader.read_file("table-125552.753236.tsv","/home/cemarks/Projects/cancer/mount_folder/input")
col = data_sheet[data_sheet.columns[0]]

annotated = score_functions.read_annotation("/home/cemarks/Projects/cancer/data/Data/training-col100_annotated/Annotated-table-125552.753236.json")
ar = score_functions.get_col_result_annotation(annotated,1)

X = build_X(col,ar,graphdb)

q = "MATCH (n:AnswerText) - [] - (m:Answer) WHERE n.name_lower='yes' RETURN ID(n),ID(m)"
o = query_graph(q,graphdb)
oo = o.values()
len(oo) # 12755


q = "MATCH (n:AnswerText) - [] - (m:Answer) RETURN ID(n),ID(m)"
o = query_graph(q,graphdb)
oo = o.values()
len(oo) # 12755


q = "MATCH (c:CDE) - [:PERMITS] - (n:Answer) - [eq:EQUALS] - (m:Concept) RETURN c.CDE_ID,eq.cde_id,c.name,n.name,m.CODE"
o = query_graph(q,graphdb)
oo = o.values()
len(oo) # 12755


q = "MATCH (n:CDE_Name) -[:IS_SHORT] - (c:CDE) WHERE n.name = \"person_gender\" RETURN ID(c),n.name"
o = query_graph(q,graphdb)
oo = o.values()


input_string = 'Not Reported'
CDE_match = oo[0][0]
q = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score ".format(str(input_string))
q += "WHERE EXISTS {{ MATCH (c:CDE) - [:PERMITS] - (a:Answer) - [:CANBE] - (n:AnswerText) WHERE ID(c) = {0:d} }} ".format(CDE_match)
q += "RETURN ID(n), n.name, score, LABELS(n)"
o = query_graph(q,graphdb)
oo = o.values()

q = "MATCH (c:CDE) - [:PERMITS] - (a:Answer) - [:CANBE] - (n:AnswerText) WHERE ID(c) = {0:d} ".format(CDE_match)
q += "RETURN c.CDE_ID, ID(n), n.name, LABELS(n)"
o = query_graph(q,graphdb)
oo = o.values()


q = "MATCH (n:Answer) - [:EQUALS] - () - [:IS_CALLED] - (M) return M.name LIMIT 1000"
o = query_graph(q,graphdb)
oo = o.values()



if is_not_nan(row['PERMISSIBLE_VALUES']):
    ans_list = answer_parser(row['PERMISSIBLE_VALUES'])
    for i in range(len(ans_list)):
        query = "MATCH " + cde_node_str + "\n"
        answer_node = "MERGE " + create_answer_str_from_list(ans_list[i],"ans"+str(i)) + "\n"
        answer_link = "CREATE (n) - [:PERMITS] -> (ans" + str(i) +")\n" 
        query += answer_node + answer_link
        for j in range(len(ans_list[i])):
            if "ncit:" in ans_list[i][j]:
                ncits = ans_list[i][j].split(" ")
                for k in range(len(ncits)):
                    ncit_list = ncits[k].split(":")
                    if len(ncit_list) > 1:
                        code = ncit_list[1]
                        query += "MERGE (concept_" + "_".join([str(i),str(j),str(k)]) + ":Concept {CODE: '" + code + "'})\n"
                        query += "MERGE (ans" + str(i) + ") - [:EQUALS] -> (concept_" + "_".join([str(i),str(j),str(k)]) + ")\n"
            else:
                answer_text_node = "MERGE " + create_answer_text_str(ans_list[i][j],"anstxt" + str(i) + "_" + str(j)) + "\n"
                can_be = "MERGE (ans" + str(i) + ") - [:CAN_BE] -> (anstxt" + str(i) + "_" + str(j) + ")\n" 
                query += answer_text_node + can_be
        print(query)
        print()



inputstr = "survivaltimeindays"

f = find_words(inputstr,synonyms)
s = synonyms['name_lower'].iloc[f]
s_list = list(set(s.loc[s.str.len() > 2]))
grps = find_exclusive_groups(inputstr,s_list)
o = []
for g in grps:
    s1,s2 = create_string(g,inputstr,s_list)
    score_vector = score_synonym_str(s1,s2,inputstr)
    o += [[(s1,s2),score_vector]]













