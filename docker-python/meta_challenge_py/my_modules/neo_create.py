#!/usr/bin/python3
import sys,os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from my_modules.data_loader import read_file
from my_modules.utils import *



def is_not_nan(whatever):
    if whatever == whatever:
        return True
    else:
        return False

def clean_field(whatever,to_lower=False):
    if is_not_nan(whatever):
        if isinstance(whatever,str):
            s = whatever.replace("\\","\\\\")
            o = s.replace("'","\\'")
            if to_lower:
                return o.lower()
            else:
                return o
        else:
            return whatever
    else:
        return None


def add_concept_from_df(row_iloc,df):
    row = df.iloc[row_iloc]
    concept_node_str = create_concept_str_from_row(row,"n")
    query = "MERGE " + concept_node_str + "\n"
    if is_not_nan(row['SYNONYMS']):
        synonyms = (row['SYNONYMS']).split("|")
        synonym_nodes = ["MERGE " + create_synonym_str(synonyms[i],"syn"+str(i)) for i in range(len(synonyms))]
        is_calleds = ["CREATE (n) - [:IS_CALLED] -> (syn" + str(i) + ")" for i in range(len(synonyms))]
        query += "\n".join(synonym_nodes) + "\n"
        query += "\n".join(is_calleds) + "\n"
    if is_not_nan(row['SEMANTIC_TYPE']):
        sts = (row['SEMANTIC_TYPE']).split("|")
        st_nodes = ["MERGE " + create_semantictype_str(sts[i],"st"+str(i)) for i in range(len(sts))]
        is_types = ["CREATE (n) - [:IS_TYPE] -> (st" + str(i) + ")" for i in range(len(sts))]
        query += "\n".join(st_nodes) + "\n"
        query += "\n".join(is_types) + "\n"
    if is_not_nan(row['PARENTS']):
        parents = (row['PARENTS']).split("|")
        for i in range(len(parents)):
            p = parents[i]
            parent_row = (df.loc[df['CODE']==p]).iloc[0]
            parent_node = create_concept_str_from_row(parent_row,"p"+str(i))
            query += "MERGE " + parent_node + "\n"
            query += "CREATE (n) - [:IS_CHILD] -> (p" + str(i) + ")" + "\n"
    return query



def create_concept_str_from_row(row,name=None):
    row_index = int(row.name)
    code = row['CODE']
    concept_name = clean_field(row['CONCEPT_NAME'])
    definition = clean_field(row['DEFINITION'])
    l = [
        f"CODE: '{code}'",
        f"CONCEPT_NAME: '{concept_name}'",
        f"DEFINITION: '{definition}'",
        f"row_index: {row_index}"
    ]
    if name is None:
        n = "(:Concept {" + ", ".join(l) + "})"
    else:
        n = "({0:s}:Concept".format(name) + " {" + ", ".join(l) + "})"
    return n


def create_synonym_str(syn_str,name=None):
    s = clean_field(syn_str)
    s_lower = s.lower()
    if name is None:
        n = f"(:Synonym {{name: '{s}', name_lower: '{s_lower}'}})"
    else:
        n = f"({name}:Synonym {{name: '{s}', name_lower: '{s_lower}'}})"
    return n

def create_semantictype_str(st_str,name=None):
    s = clean_field(st_str)
    if name is None:
        n = f"(:SemanticType {{name: '{s}'}})"
    else:
        n = f"({name}:SemanticType {{name: '{s}'}})"
    return n

def create_questiontext_str(qt_str,name=None):
    s = clean_field(qt_str)
    if name is None:
        n = f"(:QuestionText {{name: '{s}'}})"
    else:
        n = f"({name}:QuestionText {{name: '{s}'}})"
    return n


def create_cde_str_from_row(row,name=None):
    d = {
        "row_index": {
            "name": "row_index",
            "value": int(row.name),
            "type": "num"
        },
        "cde_id": {
            "name": "CDE_ID",
            "value": int(row['CDE_ID']),
            "type": "num"
        },
        "cde_long_name": {
            "name": "CDE_LONG_NAME",
            "value": clean_field(row['CDE_LONG_NAME']),
            "type": "text"
        },
        "cde_long_name_lower": {
            "name": "CDE_LONG_NAME_LOWER",
            "value": clean_field(row['CDE_LONG_NAME'],to_lower=True),
            "type": "text"
        },
        "cde_short_name": {
            "name": "name",
            "value": clean_field(row['CDE_SHORT_NAME']),
            "type": "text"
        },
        "cde_short_name_lower":{
            "name": "name_lower",
            "value": clean_field(row['CDE_SHORT_NAME'],to_lower=True),
            "type": "text"
        },
        "definition": {
            "name": "DEFINITION",
            "value": clean_field(row['DEFINITION']),
            "type": "text"
        },
        "context_name": {
            "name": "CONTEXT_NAME",
            "value": clean_field(row['CONTEXT_NAME']),
            "type": "text"
        },
        "workflow_status": {
            "name": "WORKFLOW_STATUS",
            "value": clean_field(row['WORKFLOW_STATUS']),
            "type": "text"
        },
        "registration_status": {
            "name": "REGISTRATION_STATUS",
            "value": clean_field(row['REGISTRATION_STATUS']),
            "type": "text"
        },
        "value_domain_type": {
            "name": "VALUE_DOMAIN_TYPE",
            "value": clean_field(row['VALUE_DOMAIN_TYPE']),
            "type": "text"
        },
        "datatype": {
            "name": "DATATYPE",
            "value": clean_field(row['DATATYPE']),
            "type": "text"
        },
        "unit_of_measure": {
            "name": "UNIT_OF_MEASURE",
            "value": clean_field(row['UNIT_OF_MEASURE']),
            "type": "text"
        },
        "display_format": {
            "name": "DISPLAY_FORMAT",
            "value": clean_field(row['DISPLAY_FORMAT']),
            "type": "text"
        },
        "max_value": {
            "name": "MAX_VALUE",
            "value": clean_field(row['MAX_VALUE']),
            "type": "num"
        },
        "min_value": {
            "name": "MIN_VALUE",
            "value": clean_field(row['MIN_VALUE']),
            "type": "num"
        },
        "decimal_place": {
            "name": "DECIMAL_PLACE",
            "value": clean_field(row['DECIMAL_PLACE']),
            "type": "num"
        }
    }
    l = []
    for prop in [
        "cde_id",
        "cde_long_name",
        "cde_long_name_lower",
        "cde_short_name",
        "cde_short_name_lower",
        "definition",
        "context_name",
        "workflow_status",
        "registration_status",
        "value_domain_type",
        "datatype",
        "unit_of_measure",
        "display_format",
        "max_value",
        "min_value",
        "decimal_place",
        "row_index"
    ]:
        if d[prop]['value'] is not None:
            if d[prop]['type'] == 'num':
                l.append(
                    d[prop]["name"]+": " + str(d[prop]['value'])
                )
            else:
                l.append(
                    d[prop]["name"]+": '" + str(d[prop]['value']) + "'"
                )
    if name is None:
        n = "(:CDE {" + ", ".join(l) + "})"
    else:
        n = "({0:s}:CDE".format(name) + " {" + ", ".join(l) + "})"
    return n


def concept_parser(concept_str):
    if clean_field(concept_str) is not None:
        first_split = str(clean_field(concept_str)).split("|")
        second_split = [i.split(":") for i in first_split]
        return second_split
    else:
        return None

def pipe_parser(pipe_str):
    if clean_field(pipe_str) is not None:
        first_split = str(clean_field(pipe_str)).split("|")
        return first_split
    else:
        return None

def answer_parser(ans_str):
    if clean_field(ans_str) is not None:
        first_split = str(clean_field(ans_str)).split("|")
        second_split = [i.split("\\") for i in first_split]
        return second_split
    else:
        return None

def classifications_parser(class_str):
    if clean_field(class_str) is not None:
        first_split = str(clean_field(class_str)).split("|")
        second_split = [i.split("\\") for i in first_split]
        third_split = [i[0].split(" - ") for i in second_split]
        o = [[third_split[i][0],third_split[i][1],second_split[i][1]] for i in range(len(third_split))]
        return o
    else:
        return None

def create_classification_str_from_list(class_list,name=None):
    if name is None:
        o = "(:Classification {TYPE: '" + clean_field(class_list[0]) + "', "
        o += "VALUE: '" + clean_field(class_list[1]) + "', "
        o += "CONTEXT: '" + clean_field(class_list[2]) + "'})"
    else:
        o = "(" + name + ":Classification {TYPE: '" + clean_field(class_list[0]) + "', "
        o += "VALUE: '" + clean_field(class_list[1]) + "', "
        o += "CONTEXT: '" + clean_field(class_list[2]) + "'})"
    return o

def create_answer_str_from_list(ans_list,name=None):
    if name is None:
        o = "(:Answer {name: '" + clean_field("|".join(ans_list)) + "'})"
    else:
        o = "(" + name + ":Answer {name: '" + clean_field("|".join(ans_list)) + "'})"
    return o

def create_answer_text_str(ans_text,name=None):
    clean_txt = clean_field(ans_text)
    if name is None:
        o = "(:AnswerText {name: '" + clean_txt + "', name_lower: '" + clean_txt.lower() + "'})"
    else:
        o = "(" + name + ":AnswerText {name: '" + clean_txt + "', name_lower: '" + clean_txt.lower() + "'})"
    return o


def create_dec_str_from_row(row,name=None):
    if clean_field(row['DEC_ID']) is not None:
        d = {
            "dec_id": {
                "name": "DEC_ID",
                "value": clean_field(row['DEC_ID']),
                "type": "num"
            },
            "dec_long_name": {
                "name": "name",
                "value": clean_field(row['DEC_LONG_NAME']),
                "type": "text"
            },
            "dec_long_lower": {
                "name": "name_lower",
                "value": clean_field(row['DEC_LONG_NAME'],to_lower=True),
                "type": "text"
            }
        }
        l = []
        for prop in [
            "dec_id",
            "dec_long_name",
            "dec_long_lower"
        ]:
            if d[prop]['value'] is not None:
                if d[prop]['type'] == 'num':
                    l.append(
                        d[prop]["name"]+": " + str(d[prop]['value'])
                    )
                else:
                    l.append(
                        d[prop]["name"]+": '" + str(d[prop]['value']) + "'"
                    )
        if name is None:
            n = "(:DEC {" + ", ".join(l) + "})"
        else:
            n = "({0:s}:DEC".format(name) + " {" + ", ".join(l) + "})"
        return n
    else:
        return None


def add_cde_from_df(row_iloc,df,g):
    row = df.iloc[row_iloc]
    cde_node_str = create_cde_str_from_row(row,"n")
    query = "MERGE " + cde_node_str + "\n"
    if is_not_nan(row['CDE_SHORT_NAME']):
        cde_short_names = pipe_parser(row['CDE_SHORT_NAME'])
        for i in range(len(cde_short_names)):
            query += "MERGE (cdename" + str(i) +":CDE_Name {name: '" + str(cde_short_names[i]) + "', name_lower: '" + str(cde_short_names[i]).lower() + "'})\n"
            query += "CREATE (n) - [:ISSHORT] -> (cdename" + str(i) +")\n"
    if is_not_nan(row['QUESTION_TEXT']):
        question_text = pipe_parser(row['QUESTION_TEXT'])
        for i in range(len(question_text)):
            query += "MERGE (qt" + str(i) +":QuestionText {name: '" + str(question_text[i]) + "'})\n"
            query += "CREATE (n) - [:QUESTION] -> (qt" + str(i) +")\n"
    if is_not_nan(row['DEC_ID']):
        dec_node_str = create_dec_str_from_row(row,'dec')
        query += "MERGE " + dec_node_str + "\n"
        query += "CREATE (n) - [:IS_CAT] -> (dec) \n"
        if is_not_nan(row['OBJECT_CLASS_CONCEPTS']):
            concepts = concept_parser(row['OBJECT_CLASS_CONCEPTS'])
            for i in range(len(concepts)):
                query += "MERGE (obclass" + str(i) +":Concept {CODE: '" + str(concepts[i][1]) + "'}) ON CREATE SET obclass" + str(i) + ".nonstandardtype = '" + str(concepts[i][0]) + "'\n"
                query += "CREATE (dec) - [:IS_OBJ] -> (obclass" + str(i) +")\n"
    if is_not_nan(row['PROPERTY_CONCEPTS']):
            concepts = concept_parser(row['PROPERTY_CONCEPTS'])
            for i in range(len(concepts)):
                query += "MERGE (prop" + str(i) +":Concept {CODE: '" + str(concepts[i][1]) + "'}) ON CREATE SET prop" + str(i) + ".nonstandardtype = '" + str(concepts[i][0]) + "'\n"
                query += "CREATE (dec) - [:IS_PROP] -> (prop" + str(i) +")\n"
    if is_not_nan(row['CLASSIFICATIONS']):
        classes = classifications_parser(row['CLASSIFICATIONS'])
        class_nodes = ["MERGE " + create_classification_str_from_list(classes[i],"cl"+str(i)) for i in range(len(classes))]
        is_classes = ["CREATE (n) - [:IS_CLASS] -> (cl" + str(i) + ")" for i in range(len(classes))]
        query += "\n".join(class_nodes) + "\n"
        query += "\n".join(is_classes) + "\n"
    with g.session() as q:
        z = q.run(query)
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
                            query += "CREATE (ans" + str(i) + ") - [eq" + str(i) +":EQUALS { cde_id: n.CDE_ID }] -> (concept_" + "_".join([str(i),str(j),str(k)]) + ")\n"
                else:
                    answer_text_node = "MERGE " + create_answer_text_str(ans_list[i][j],"anstxt" + str(i) + "_" + str(j)) + "\n"
                    can_be = "MERGE (ans" + str(i) + ") - [:CANBE] -> (anstxt" + str(i) + "_" + str(j) + ")\n" 
                    query += answer_text_node + can_be
        with g.session() as q:
            z = q.run(query)
    return True

def delete_everything(g):
    with g.session() as q:
        z = q.run("MATCH (n) - [r] - () DELETE n,r")
        z = q.run("MATCH (n) DELETE n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        REF_DIR = sys.argv[1]
    else:
        REF_DIR = "/data"
    t = read_file("Thesaurus.tsv",REF_DIR)
    c = read_file("caDSR-export-20190528-1320.tsv",REF_DIR)
    graphDB = neo4j_connect()
    with graphDB.session() as q:
        q.run("CREATE CONSTRAINT concept_code ON (concept:Concept) ASSERT concept.CODE IS UNIQUE")
        q.run("CREATE CONSTRAINT synonym_name ON (synonym:Synonym) ASSERT synonym.name IS UNIQUE")
        q.run("CREATE CONSTRAINT st_name ON (st:SemanticType) ASSERT st.name IS UNIQUE")
        # q.run("CREATE CONSTRAINT cde_id ON (cde:CDE) ASSERT cde.ID IS UNIQUE")
        q.run("CREATE CONSTRAINT dec_id ON (dec:DEC) ASSERT dec.ID is UNIQUE")
    for row_iloc in range(0,len(t)):
        query_str = add_concept_from_df(row_iloc,t)
        with graphDB.session() as q:
            z = q.run(query_str)
        if row_iloc % 1000 == 0:
            print("{0:d}/{1:d}".format(row_iloc,len(t)))
    for row_iloc in range(0,len(c)):
        z = add_cde_from_df(row_iloc,c,graphDB)
        if row_iloc % 100 == 0:
            print("{0:d}/{1:d}".format(row_iloc,len(c)))
    # fulltext indices
    with graphDB.session() as q:
        i0 = q.run("CALL db.index.fulltext.createNodeIndex(\"ansindex\",[\"AnswerText\"],[\"name\"])")
        i1 = q.run("CALL db.index.fulltext.createNodeIndex(\"nameindex\",[\"Synonym\",\"CDE\",\"DEC\",\"CDE_Name\",\"QuestionText\"],[\"name\",\"CDE_LONG_NAME\"])")
    graphDB.close()
