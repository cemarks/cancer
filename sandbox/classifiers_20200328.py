#!/usr/bin/python3

import pandas as pd
from my_modules.utils import *
from my_modules import score_functions
import numpy as np
import re
import stringdist

FT_SEARCH_CUTOFF = np.exp(1) # Discard search results with lower scores than this
NAMEINDEX_SEARCH_REQD = 25 # If this many results are not found in full text searches on column names, a synonym decomposition will be used.
NAMEINDEX_CDE_REQD = 5
FOLLOW_ON_SEARCH_MIN_WORD_LEN = 3

X_FT_STRUCTURE = {
    'index':{
        'column_no': 0,
        'ft_postprocess_params': {}
    },
    'ftsearch_syn_class':{
        'column_no': 1,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_CLASS] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_syn_prop':{
        'column_no': 2,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_PROP] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_syn_obj':{
        'column_no': 3,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_OBJ] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_cde':{
        'column_no': 4,
        'ft_postprocess_params': {
            'CDE_Name':{
                'query': lambda z: f"MATCH (n) - [:IS_SHORT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: max([old,new])
            },
            'CDE':{
                'query': lambda z: f"RETURN {z}",
                'aggregation': lambda old,new: max([old,new])
            }
        }
    },
    'ftsearch_dec':{
        'column_no': 5,
        'ft_postprocess_params': {
            'DEC':{
                'query': lambda z: f"MATCH (n) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_question':{
        'column_no': 6,
        'ft_postprocess_params': {
            'QuestionText':{
                'query': lambda z: f"MATCH (n) - [:QUESTION] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: max([old,new])
            }
        }
    },
    # 'synsearch_class':{
    #     'column_no': 10,
    #     'apply_to':[
    #         'syn_search'
    #     ],
    #     'synsearch_postprocess_params': {
    #         'Synonym':{
    #             'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_CLASS] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
    #             'aggregation': lambda old,new: old + new
    #         }
    #     }
    # },
    # 'synsearch_prop':{
    #     'column_no': 11,
    #     'apply_to':[
    #         'syn_search'
    #     ],
    #     'synsearch_postprocess_params': {
    #         'Synonym':{
    #             'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_PROP] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
    #             'aggregation': lambda old,new: old + new
    #         }
    #     }
    # },
    # 'synsearch_obj':{
    #     'column_no': 12,
    #     'apply_to':[
    #         'syn_search'
    #     ],
    #     'synsearch_postprocess_params': {
    #         'Synonym':{
    #             'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_OBJ] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
    #             'aggregation': lambda old,new: old + new
    #         }
    #     }
    # }
}

def check_datatype(value,data_type_str):
    check = False
    if data_type_str in [
        'DATETIME',
        'SAS Date',
        'SAS Time',
        'DATE',
        'ALPHANUMERIC',
        'TIME',
        'HL7EDv3',
        'BOOLEAN',
        'Numeric Alpha DVG',
        'Date Alpha DVG',
        'Derived',
        'UMLUidv1.0',
        'DATE/TIME',
        'HL7STv3',
        'CLOB',
        'HL7CDv3',
        'HL7TSv3',
        'HL7PNv3',
        'HL7TELv3',
        'OBJECT',
        'Alpha DVG'
    ]:
        check = True
    elif data_type_str in ["CHARACTER",'varchar']:
        if re.fullmatch(r"[0-9]*.{0,1}[0-9]*",value):
            return False
        else:
            return True
    elif data_type_str in [
        'Integer',
        'HL7INTv3'
    ]:
        try:
            r = float(value)
            i = int(r)
            if r == i:
                check = True
        except ValueError as e:
            check = False
    elif data_type_str in [
        'NUMBER',
        'HL7REALv3'
    ]:
        try:
            r = float(value)
            check = True
        except ValueError as e:
            check = False
    elif data_type_str == 'binary':
        try:
            r = float(value)
            i = int(r)
            if i in [0,1]:
                check = True
        except ValueError as e:
            check = False
    else:
        check = True
    return check


def check_date_num(month,day):
    check = False
    if (month==2) and (day > 0) and (day < 30):
        check = True
    if (month in [4,6,9,11]) and (day > 0) and (day < 31):
        check = True
    if (month in [1,3,5,7,8,10,12]) and (day > 0) and (day < 32):
        check = True
    return check

def check_hr(hr_int,hr24=False):
    check = False
    if hr24:
        if (hr_int >= 0) and (hr_int <= 24):
            check = True
    else:
        if (hr_int >= 0) and (hr_int <= 12):
            check = True
    return check

def check_min_sec(input_int):
    check = False
    if (input_int >= 0) and (input_int <= 59):
        check = True
    return check

def check_display_format(input_str,display_format):
    check = False
    if display_format in [
        'mm/dd/yy'
    ]: 
        if re.fullmatch(r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{2}',input_str):
            s = input_str.split("/")
            try:
                m = int(s[0])
                d = int(s[1])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == 'DD/MON/YYYY': 
        if re.fullmatch(r'[0-9]{1,2}/[A-Z,a-z]{3}/[0-9]{4}',input_str):
            s = input_str.split("/")
            m_str = s[1].lower()
            if m_str == 'jan':
                m = 1
            elif m_str == 'feb':
                m = 2
            elif m_str == 'mar':
                m = 3
            elif m_str == 'apr':
                m = 4
            elif m_str == 'may':
                m = 5
            elif m_str == 'jun':
                m = 6
            elif m_str == 'jul':
                m = 7
            elif m_str == 'aug':
                m = 8
            elif m_str == 'sep':
                m = 9
            elif m_str == 'oct':
                m = 10
            elif m_str == 'nov':
                m = 11
            elif m-str == 'dec':
                m = 12
            else:
                m = 13
            d = int(s[0])
            check = check_date_num(m,d)
    elif display_format == 'YYYY-MM-DD':
        if re.fullmatch(r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}',input_str):
            s = input_str.split("-")
            try:
                m = int(s[1])
                d = int(s[2])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == 'YYYY':
        if re.fullmatch(r'[0-9]{4}',input_str):
            check = True
    elif display_format == 'TIME (HR(24):MN':
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
            except ValueError as e:
                h = 25
                m = 61
            if check_hr(h,hr24=True) and check_min_sec(m):
                check = True
    elif display_format == 'YYYYMMDD':
        if re.fullmatch(r'[0-9]{8}',str(input_str)):
            try:
                m = int(str(input_str)[4:6])
                d = int(str(input_str)[6:8])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == '9999.99':
        if re.fullmatch(r'[0-9]{1,4}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,4}',str(input_str)):
            check = True
    elif display_format in [
        'mm/dd/yyyy',
        'MM/DD/YYYY'
    ]: 
        if re.fullmatch(r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}',input_str):
            s = input_str.split("/")
            try:
                m = int(s[0])
                d = int(s[1])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == '9999999':
        if re.fullmatch(r'[0-9]{1,7}',str(input_str)):
            check = True
    elif display_format == '10,3':
        if re.fullmatch(r'[0-9]{1,10}\.[0-9]{0,3}',str(input_str)) or re.fullmatch(r'[0-9]{1,10}',str(input_str)):
            check = True
    elif display_format == '9999.9':
        if re.fullmatch(r'[0-9]{1,4}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,4}',str(input_str)):
            check = True
    elif display_format == '%':
        if re.fullmatch(r'[0-9]*.{0,1}[0-9]*\%',str(input_str)) or re.fullmatch(r'[0-9]*.{0,1}[0-9]*',str(input_str)):
            check = True
    elif display_format == '999.9':
        if re.fullmatch(r'[0-9]{1,3}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,3}',str(input_str)):
            check = True
    elif display_format == '99.9':
        if re.fullmatch(r'[0-9]{1,2}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,2}',str(input_str)):
            check = True
    elif display_format == 'hh:mm:ss':
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
                sec = int(s[2])
            except ValueError as e:
                h = 25
                m = 61
                sec = 61
            if check_hr(h,hr24=True) and check_min_sec(m) and check_min_sec(sec):
                check = True
    elif display_format == '999.99':
        if re.fullmatch(r'[0-9]{1,3}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,3}',str(input_str)):
            check = True
    elif display_format == '9.999':
        if re.fullmatch(r'[0-9]{0,1}\.[0-9]{0,3}',str(input_str)) or re.fullmatch(r'[0-9]{1}',str(input_str)):
            check = True
    elif display_format == '999999.9':
        if re.fullmatch(r'[0-9]{0,6}\.[0-9]{0,1}',str(input_str)) or re.fullmatch(r'[0-9]{1,6}',str(input_str)):
            check = True
    elif display_format in [
        'hh:mm',
        'TIME_HH:MM'
    ]:
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
            except ValueError as e:
                h = 25
                m = 61
            if check_hr(h,hr24=True) and check_min_sec(m):
                check = True
    elif display_format == 'hh:mm:ss:rr':
        if re.fullmatch(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}:[0-9]{1,2}',input_str):
            s = input_str.split(":")
            try:
                h = int(s[0])
                m = int(s[1])
                sec = int(s[2])
            except ValueError as e:
                h = 25
                m = 61
                sec = 61
            if check_hr(h,hr24=True) and check_min_sec(m) and check_min_sec(sec):
                check = True
    elif display_format == 'TIME_MIN':
        check = True
    elif display_format == '99.99':
        if re.fullmatch(r'[0-9]{0,2}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,2}',str(input_str)):
            check = True
    elif display_format == '9999.999':
        if re.fullmatch(r'[0-9]{0,4}\.[0-9]{0,3}',str(input_str)) or re.fullmatch(r'[0-9]{1,4}',str(input_str)):
            check = True
    elif display_format == 'MMYYYY':
        if re.fullmatch(r'[0-9]{6}',str(input_str)):
            try:
                m = int(str(input_str)[0:2])
            except ValueError as e:
                m = 13
            check = check_date_num(m,1)
    elif display_format == '99999.99':
        if re.fullmatch(r'[0-9]{0,5}\.[0-9]{0,2}',str(input_str)) or re.fullmatch(r'[0-9]{1,5}',str(input_str)):
            check = True
    elif display_format == 'MMDDYYYY':
        if re.fullmatch(r'[0-9]{6}',str(input_str)):
            try:
                m = int(str(input_str)[0:2])
                d = int(str(input_str)[2:4])
            except ValueError as e:
                m = 13
                d = 32
            check = check_date_num(m,d)
    elif display_format == '999-99-9999':
        if re.fullmatch(r'[0-9]{3}-[0-9]{2}-[0-9]{4}',str(input_str)):
            check = True
    elif display_format == 'YYYYMM':
        if re.fullmatch(r'[0-9]{6}',str(input_str)):
            try:
                m = int(str(input_str)[4:6])
            except ValueError as e:
                m = 13
            check = check_date_num(m,1)
    else:
        check = True
    return check




def find_synonyms(input_str,g):
    query = "MATCH (n:Synonym) WHERE \"{0:s}\" CONTAINS n.name_lower RETURN n.name_lower".format(str(input_str).lower())
    result = query_graph(query,g)
    values = result.value()
    return values


def full_str_match_synonym(col_name,g):
    string_list = expand_string_names(col_name)
    where_clause_list = ["n.lower = '" + i + "'" for i in string_list]
    where_clause = " OR ".join(where_clause_list)
    query = "MATCH (n:Synonym) WHERE " + where_clause + "RETURN n.name"
    result = query_graph(query,g)
    return result.value()

def full_str_match_short(col_name,g):
    string_list = expand_string_names(col_name)
    where_clause_list = ["n.cde_short_name_lower = '" + i + "'" for i in string_list]
    where_clause = " OR ".join(where_clause_list)
    query = "MATCH (n:CDE) WHERE " + where_clause + "RETURN n.name"
    result = query_graph(query,g)
    return result.value()

def nameindex_query(input_string,g):
    query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node, score RETURN ID(node), score, LABELS(node)".format(str(input_string))
    result = query_graph(query,g)
    return result.values()

def create_data_row(input_index):
    cols = [(i,X_FT_STRUCTURE[i]['column_no']) for i in X_FT_STRUCTURE]
    cols.sort(key=lambda z: z[1])
    d = {i[0]:[input_index] if i[0]=='index' else [0] for i in cols}
    return pd.DataFrame(d)

def get_CDE_indices(ft_result,update_column,g):
    node_type = ft_result[2][0]
    node_index = ft_result[0]
    query = X_FT_STRUCTURE[update_column]['ft_postprocess_params'][node_type]['query'](node_index)
    result = query_graph(query,g)
    values = result.value()
    return values

def update_data(df,ft_result,update_column,update_index,g):
    plus_value = ft_result[1]
    node_type = ft_result[2][0]
    update_indices = get_CDE_indices(ft_result,update_column,g)
    if update_column not in df.columns:
        raise ValueError("{0:s} column not in DataFrame".format(update_column))
    new_value = X_FT_STRUCTURE[update_column]['ft_postprocess_params'][node_type]['aggregation'](
        df[update_column].loc[df['index']==update_index].values[0],
        plus_value
    )
    df.loc[df['index']==update_index,update_column] = new_value


def create_or_update(df,ft_result,update_column,g):
    update_indices = get_CDE_indices(ft_result,update_column,g)
    for update_index in update_indices:
        if update_column not in df.columns:
            raise ValueError("{0:s} column not in DataFrame".format(update_column))
        if update_index not in df['index'].values:
            new_row = create_data_row(update_index)
            df = df.append(new_row)
        update_data(df,ft_result,update_column,update_index,g)
    return df

# def enumeration_exact_search(unique_values_list,g):
#     unique_values_lower = ["'" + str(i).lower() + "'" for i in unique_values_list]
#     unique_values_lower = list(set(unique_values_lower))
#     query = "MATCH (n:AnswerText) - [:CAN_BE] - (m:Answer) WHERE n.name_lower in [{0:s}] ".format(",".join(unique_values_lower))
#     query += "WITH DISTINCT(m) AS ans MATCH (ans) - [:PERMITS] - (c:CDE) "
#     query += "RETURN ID(c), c.CDE_ID, COUNT(*)*1.0/{0:d}".format(len(unique_values_list))
#     result = query_graph(query,g)
#     values = result.values()
#     values.sort(key = lambda z: z[2], reverse=True)
#     return values

def enumeration_exact_search(unique_values_list,g):
    unique_values_lower = ["'" + str(i).lower() + "'" for i in unique_values_list]
    unique_values_lower = list(set(unique_values_lower))
    query = "MATCH (n:AnswerText) - [:CAN_BE] - (m:Answer) - [:PERMITS] - (c:CDE) WHERE n.name_lower in [{0:s}] ".format(",".join(unique_values_lower))
    query += "WITH DISTINCT n.name_lower AS name_lower, c AS c_distinct "
    query += "RETURN ID(c_distinct), c_distinct.CDE_ID, COUNT(*)*1.0/{0:d}".format(len(unique_values_list))
    result = query_graph(query,g)
    values = result.values()
    values.sort(key = lambda z: z[2], reverse=True)
    return values


# def enumeration_fulltext_search(unique_values_list,g):
#     query = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
#     query += " MATCH (n:AnswerText) - [:CAN_BE] - (m:Answer) - [:PERMITS] - (c:CDE)\n"
#     query += " RETURN ID(c), c.CDE_ID, score"
#     result = query_graph(query,g)
#     values = result.values()
#     values.sort(key = lambda z: z[2], reverse=True)
#     return values


# def enumeration_ansindex_single_search(single_unique_value,g):
#     query = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
#     query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer)\n"
#     query += " WITH a,MAX(score) AS max_score\n"
#     query += " MATCH (a:Answer) - [:PERMITS] - (c:CDE)\n"
#     query += " RETURN ID(c), c.CDE_ID, SUM(max_score)"
#     result = query_graph(query,g)
#     return result.values()

def enumeration_ansindex_single_search(single_unique_value,g):
    query = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
    query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer) - [:PERMITS] - (c:CDE) \n"
    query += " RETURN ID(c), c.CDE_ID, MAX(score)"
    result = query_graph(query,g)
    return result.values()


# def enumeration_concept_single_search(single_unique_value,g):
#     query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
#     query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer)\n"
#     query += " WITH a,MAX(score) AS max_score\n"
#     query += " MATCH (a) - [:PERMITS] - (c:CDE)\n"
#     query += " RETURN ID(c), c.CDE_ID, SUM(max_score)"
#     result = query_graph(query,g)
#     values = result.values()
#     values.sort(key = lambda z: z[2], reverse=True)
#     return values

def enumeration_concept_single_search(single_unique_value,g):
    query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
    query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
    query += " RETURN ID(c), c.CDE_ID, MAX(score)"
    result = query_graph(query,g)
    values = result.values()
    values.sort(key = lambda z: z[2], reverse=True)
    return values


def enumeration_concept_search(value_list,g):
    value_set = list(set(value_list))
    query = "CALL {\n"
    v = value_set[0]
    query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
    query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
    query += " RETURN ID(c) AS cde_index, c.CDE_ID AS cde_id, MAX(score) AS max_score \n"
    for v in value_set[1:len(value_list)]:
        query += " UNION ALL\n"
        query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
        query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
        query += " RETURN ID(c) AS cde_index, c.CDE_ID AS cde_id, MAX(score) AS max_score \n"
    query += "}\n"
    query += "RETURN cde_index, cde_id, SUM(max_score) * 1.0 / {0:d}".format(len(value_set))
    result = query_graph(query,g)
    values = result.values()
    values.sort(key = lambda z: z[2], reverse=True)
    return values

# def enumeration_ansindex_search(value_list,g):
#     value_set = list(set(value_list))
#     query = "CALL {\n"
#     v = value_set[0]
#     query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
#     query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer)\n"
#     query += " WITH a,MAX(score) AS max_score\n"
#     query += " MATCH (a) - [:PERMITS] - (c:CDE)\n"
#     query += " RETURN ID(c) AS cde_index, c.CDE_ID AS cde_id, SUM(max_score) AS sum_max_score \n"
#     for v in value_set[1:len(value_list)]:
#         query += " UNION ALL\n"
#         query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
#         query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer)\n"
#         query += " WITH a,MAX(score) AS max_score\n"
#         query += " MATCH (a) - [:PERMITS] - (c:CDE)\n"
#         query += " RETURN ID(c) AS cde_index, c.CDE_ID AS cde_id, SUM(max_score) AS sum_max_score \n"
#     query += "}\n"
#     query += "RETURN cde_index, cde_id, SUM(sum_max_score) * 1.0 / {0:d}".format(len(value_set))
#     result = query_graph(query,g)
#     values = result.values()
#     values.sort(key = lambda z: z[2], reverse=True)
#     return values

def enumeration_ansindex_search(value_list,g):
    value_set = list(set(value_list))
    query = "CALL {\n"
    v = value_set[0]
    query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
    query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
    query += " RETURN ID(c) AS cde_index, c.CDE_ID AS cde_id, MAX(score) AS max_score \n"
    for v in value_set[1:len(value_list)]:
        query += " UNION ALL\n"
        query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
        query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
        query += " RETURN ID(c) AS cde_index, c.CDE_ID AS cde_id, MAX(score) AS max_score \n"
    query += "}\n"
    query += "RETURN cde_index, cde_id, SUM(max_score) * 1.0 / {0:d}".format(len(value_set))
    result = query_graph(query,g)
    values = result.values()
    values.sort(key = lambda z: z[2], reverse=True)
    return values


def score_enumeration_check(column_series):
    no_nan = column_series.loc[column_series == column_series]
    l_all = len(no_nan)
    l = len(no_nan.unique())
    if (l < (np.sqrt(l_all))) and (str(no_nan.dtype)[0] not in ['f','i','u']):
        return True
    else:
        return False

def find_exclusive_groups(input_str,word_list):
    groups = [[i] for i in range(len(word_list))]
    i = 0
    while i < len(groups):
        g = groups[i]
        larger_groups = get_larger_groups(g,input_str,word_list)
        if len(larger_groups) > 0:
            # lg = [l for l in larger_groups if not any([all([j in k for j in l]) for k in groups])]
            groups = groups[0:i] + larger_groups + groups[(i+1):len(groups)]
        else:
            if any([all([j in k for j in g]) for k in groups[0:i] + groups[(i+1):len(groups)]]):
                groups = groups[0:i] + groups[(i+1):len(groups)]
            else:
                i+=1
    return groups

def get_larger_groups(group_inds,input_str,word_list,larger_inds_only=True):
    word = input_str
    for i in group_inds:
        word = word.replace(word_list[i]," ")
    larger_groups = []
    other_inds = list(set(range(len(word_list))) - set(group_inds))
    if larger_inds_only:
        other_inds = [i for i in other_inds if i > max(group_inds)]
    for i in other_inds:
        if word_list[i] in word:
            larger_groups.append(group_inds + [i])
    return larger_groups


def create_string(group_inds,input_str,word_list):
    str_list = [input_str]
    words = [word_list[i] for i in group_inds]
    words.sort(key=lambda z: len(z), reverse=True)
    for i in range(len(words)):
        new_s = []
        w = words[i]
        for j in range(len(str_list)):
            s = str_list[j]
            if s in words[0:i]:
                new_s.append(s)
            else:
                matches = re.finditer(words[i],s)
                k = 0
                for m in matches:
                    if m.start() > k:
                        new_s.append(s[k:m.start()])
                    new_s.append(s[m.start():m.end()])
                    k = m.end()
                if k < len(s):
                    new_s.append(s[k:len(s)])
        str_list = new_s
    str_list_words_only = [i for i in str_list if i in words]
    return " ".join(str_list)," ".join(str_list_words_only)

def score_synonym_str(str1_full,str2_wordsonly,input_str):
    pct_coverage = len(str2_wordsonly.replace(" ",""))/len(input_str)
    space_count = str1_full.count(" ")
    out_word_count = space_count - str2_wordsonly.count(" ")
    word_lengths = [len(i) for i in str1_full.split(" ")]
    out_word_lengths = [len(i) for i in str1_full.split(" ") if i not in str2_wordsonly.split(" ")]
    score = pct_coverage * (1-(space_count+1)/(len(input_str)))
    return score

def create_new_search_strings(input_str,g):
    word_list = find_synonyms(input_str,g)
    word_list = [w for w in word_list if len(w) >= FOLLOW_ON_SEARCH_MIN_WORD_LEN]
    exclusive_groups = find_exclusive_groups(input_str,word_list)
    strings = list(set([create_string(i,input_str,word_list) for i in exclusive_groups]))
    scores = [score_synonym_str(i[0],i[1],input_str) for i in strings]
    combined_list = [[strings[i],scores[i]] for i in range(len(strings))]
    combined_list.sort(key=lambda z: z[1], reverse=True)
    return [(i[0][0],i[1]) for i in combined_list[0:3]]


def get_enum_answers(cde_index,g):
    query = "MATCH (n:CDE) - [:PERMITS] - (a:Answer) - [:CAN_BE] - (at: AnswerText) WHERE ID(n) = {0:d} RETURN ID(at),at.name".format(cde_index)
    result = query_graph(query,g)
    values = result.values()
    return values


def is_enum(col_series):
    no_nan = col_series.loc[col_series == col_series]
    u = no_nan.unique()
    if len(u) < np.sqrt(len(no_nan)):
        return True
    else:
        return False

def nans_vs_nexp(n_ans,n_exp):
    return 1-(np.abs(n_ans-n_exp)/(n_ans + n_exp))

def create_answer_count_df(cde_indices,g):
    q = "MATCH (n:CDE) - [:PERMITS] - (m:Answer) WHERE ID(n) IN [{0:s}] RETURN ID(n),COUNT(*)".format(",".join(cde_indices.astype('int').astype('str')))
    result = query_graph(q,g)
    answer_counts = result.values()
    q2 = "MATCH (n:CDE) WHERE ID(n) IN [{0:s}] AND n.DATATYPE = 'BOOLEAN' RETURN ID(n),2".format(",".join(cde_indices.astype('int').astype('str')))
    result = query_graph(q2,g)
    answer_counts2 = result.values()
    n1 = [i[0] for i in answer_counts]
    answer_counts2 = [i for i in answer_counts2 if i[0] not in n1]
    answer_count_df = pd.DataFrame(answer_counts + answer_counts2,columns=['index','answer_count'])
    return answer_count_df




def build_X(col_series,annotated_result,g):
    t = time.time()
    col_name = col_series.name
    search_string = lower_upper_split(
        period_replace(
            underscore_replace(
                col_name
            )
        )
    )
    df = pd.DataFrame(columns = [i for i in X_FT_STRUCTURE])
    result_types = list(set([j for i in X_FT_STRUCTURE for j in X_FT_STRUCTURE[i]['ft_postprocess_params']]))
    result_type_dict = {i:[j for j in X_FT_STRUCTURE if i in X_FT_STRUCTURE[j]['ft_postprocess_params']] for i in result_types}
    search_results = nameindex_query(search_string,g)
    print("Search Results Complete: {0:d} sec".format(int(time.time() - t)))
    print("{0:d} results.\n".format(len(search_results)))
    data_adds = list(set([i[0] for i in search_results if i[1] > FT_SEARCH_CUTOFF]))
    search_results_add = [i for i in search_results if i[0] in data_adds]
    if (len(search_results) < NAMEINDEX_SEARCH_REQD) or (len([i for i in search_results if i[2][0] in ['CDE','CDE_Name','DEC','QuestionText']]) < NAMEINDEX_CDE_REQD):
        new_search_strings = create_new_search_strings(search_string,g)
        for s in new_search_strings:
            search_results = nameindex_query(s[0],g)
            search_array = np.array(search_results)
            search_array[:,1] = s[1] * search_array[:,1]
            search_results = search_array.tolist()
            new_data_adds = list(set([i[0] for i in search_results if (i[1] > FT_SEARCH_CUTOFF) and (i[0] not in data_adds)]))
            data_adds += new_data_adds
            search_results_add += [i for i in search_results if i[0] in new_data_adds]
    for sr in search_results_add:
        update_columns = result_type_dict[sr[2][0]]
        for uc in update_columns:
            df = create_or_update(df,sr,uc,g)
    print("Initial DF built: {0:d} sec".format(int(time.time() - t)))
    print("DF size: {0:d}.\n".format(df.shape[0]))
    no_nan = col_series.loc[col_series==col_series]
    unique_values = no_nan.unique().tolist()
    enum_results = enumeration_exact_search(unique_values,g)
    enum_score_df = pd.DataFrame([[i[0],max([1,i[2]])] for i in enum_results],columns=['index','enum_search_score'])
    print("Exact Enum Results Complete: {0:d} sec".format(int(time.time() - t)))
    print("Enum DF size: {0:d}.\n".format(enum_score_df.shape[0]))
    df = pd.merge(df,enum_score_df,on='index',how='outer')
    df = df[[i for i in X_FT_STRUCTURE] + ['enum_search_score']]
    print("Exact Enum Results Complete: {0:d} sec".format(int(time.time() - t)))
    print("DF size: {0:d}.\n".format(df.shape[0]))
    cde_indices = df['index'].values
    answer_count_df = create_answer_count_df(cde_indices,g)
    answer_count_df = pd.merge(df['index'],answer_count_df, on='index',how='outer')
    print("Answer Count DF Created: {0:d} sec".format(int(time.time() - t)))
    print("Answer Count DF size: {0:d}.\n".format(answer_count_df.shape[0]))
    n_ans = len(unique_values)
    n_lines = len(no_nan)
    answer_count_df['answer_count'].loc[answer_count_df['answer_count'] != answer_count_df['answer_count']] = n_lines
    if answer_count_df.shape[0] > 0:
        enum_score2_df = pd.DataFrame(
            {
                'index': answer_count_df['index'],
                'enum_count_score': answer_count_df.apply(lambda x: nans_vs_nexp(n_ans,x[1]), axis=1)
            }
        )
    else:
        enum_score2_df = pd.DataFrame(
            {
                'index': [],
                'enum_count_score': []
            }
        )
    print("Enum Score2 DF Created: {0:d} sec".format(int(time.time() - t)))
    print("Enum Score2 DF size: {0:d}.\n".format(enum_score2_df.shape[0]))
    df = pd.merge(df,enum_score2_df,on='index',how='inner')
    for c in df.columns:
        v = df[c] != df[c]
        if any(v):
            df.loc[v,c] = 0
    print("DF updated: {0:d} sec".format(int(time.time() - t)))
    print("DF size: {0:d}.\n".format(df.shape[0]))
    value_domains = {i:classify_values(unique_values,i,g) for i in df['index']}
    value_domain_est = pd.DataFrame(
        {
            'index': df['index'],
            'value_domain_est': df.apply(
                lambda x: score_value_match(
                    value_domains[x[0]][0],
                    x[0],
                    g
                ),
                axis = 1
            )
        }
    )
    print("Value domain est DF Created: {0:d} sec".format(int(time.time() - t)))
    print("Enum Score2 DF size: {0:d}.\n".format(value_domain_est.shape[0]))
    df = pd.merge(df,value_domain_est,on='index',how='inner')
    print("DF updated: {0:d} sec".format(int(time.time() - t)))
    print("DF size: {0:d}.\n".format(df.shape[0]))
    annotated_cde = int(score_functions.get_de_id(annotated_result))
    metrics = pd.DataFrame(
        {
            'index': df['index'],
            'metric1': [score_functions.WEIGHTS['de_wt'] * score_functions.score_cde(annotated_cde,value_domains[i][1]) for i in df['index']],
            'metric2': [score_functions.WEIGHTS['dec_wt'] *score_functions.score_concept_overlap(annotated_cde,value_domains[i][1],g) for i in df['index']],
            'metric3': [score_functions.WEIGHTS['vd_wt'] *score_functions.score_value_domain(value_domains[i][0],annotated_result,g) for i in df['index']]
        }
    )
    print("Metric DF created: {0:d} sec".format(int(time.time() - t)))
    print("Metric DF size: {0:d}.\n".format(metrics.shape[0]))
    df = pd.merge(df,metrics,on='index',how='inner')
    df['index'] = df['index'].astype('int')
    print("DF updated: {0:d} sec".format(int(time.time() - t)))
    print("DF size: {0:d}.\n".format(df.shape[0]))
    return df

def classify_values(col_values,cde_index,g):
    query = "MATCH (n:CDE) where ID(n) = {0:d} RETURN n.DATATYPE, n.VALUE_DOMAIN_TYPE, n.DISPLAY_FORMAT, n.CDE_ID".format(int(cde_index))
    result = query_graph(query,g)
    values = result.values()
    value_domain_attributes = values[0]
    output_list = []
    if value_domain_attributes[1]=='Enumerated':
        for v in col_values:
            classification_dict = classify_enum_value(v,int(cde_index),g)
            output_list.append(classification_dict)
    elif value_domain_attributes[2] is not None:
        for v in col_values:
            classification_dict = classify_display_value(v,value_domain_attributes[2])
            output_list.append(classification_dict)
    else:
        for v in col_values:
            classification_dict = classify_datatype(v,value_domain_attributes[2])
            output_list.append(classification_dict)
    return (output_list,value_domain_attributes[3])

def classify_enum_value(col_value,cde_index,g):
    output_dict = {'observedValue':str(col_value),'permissibleValue':{}}
    query = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node as a, score ".format(str(col_value))
    query += "MATCH (n:CDE) - [:PERMITS] - (ans:Answer) - [:CAN_BE] - (a:AnswerText) WHERE ID(n) = {0:d} ".format(cde_index)
    query += "RETURN ID(ans), 'Answer', score, a.name"
    result = query_graph(query,g)
    answer_values = result.values()
    query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node as s, score ".format(str(col_value))
    query += "MATCH (n:CDE) - [:PERMITS] - (ans:Answer) - [:EQUALS] - (con:Concept) - [:IS_CALLED] - (s:Synonym) WHERE ID(n) = {0:d} ".format(cde_index)
    query += "RETURN ID(ans), 'Synonym', score, s.name, con.CODE"
    result = query_graph(query,g)
    syn_values = result.values()
    all_results = answer_values + syn_values
    all_results = [i for i in all_results if i[2] > FT_SEARCH_CUTOFF]
    if len(all_results) > 0:
        all_results.sort(key=lambda z: z[2],reverse=True)
        ans_index = all_results[0][0]
        ans_results = [i for i in all_results if i[0] == ans_index]
        # Now we need to choose the best synonym
        synonyms = [i for i in ans_results if i[1] == 'Synonym']
        if len(synonyms) > 0:  #Choose the best based on 1: search score, and 2: stringdist
            synonyms.sort(key=lambda z: (-z[2],stringdist.levenshtein_norm(z[3],col_value)))
            output_dict['permissibleValue']['value'] = str(synonyms[0][3])
            output_dict['permissibleValue']['conceptCode'] = 'ncit:' + str(synonyms[0][4])
        else:
            query = "MATCH (a:Answer) - [:EQUALS] - (c:Concept) - [:IS_CALLED] - (s:Synonym) WHERE ID(a) = {0:d} RETURN c.CODE,s.name".format(ans_index)
            result = query_graph(query,g)
            values = result.values()
            if len(values) > 0:
                values.sort(key = lambda z: stringdist.levenshtein_norm(str(col_value).lower(),str(z[1]).lower()))
                output_dict['permissibleValue']['value'] = str(values[0][1])
                output_dict['permissibleValue']['conceptCode'] = 'ncit:' + str(values[0][0])
            else:
                output_dict['permissibleValue']['value'] = str(ans_results[0][3])
                output_dict['permissibleValue']['conceptCode'] = None
    else:
        output_dict['permissibleValue']['value'] = 'NOMATCH'
        output_dict['permissibleValue']['conceptCode'] = None
    return output_dict


def classify_display_value(col_value,display_format):
    check = check_display_format(str(col_value),str(display_format))
    if check:
        output_value = "CONFORMING"
    else:
        output_value = "NONCONFORMING"
    output_dict = {
        'observedValue':str(col_value),
        'permissibleValue':{
            'value': output_value,
            'conceptCode': None
        }
    }
    return output_dict

def classify_datatype(col_value,datatype):
    check = check_datatype(str(col_value),str(datatype))
    if check:
        output_value = "CONFORMING"
    else:
        output_value = "NONCONFORMING"
    output_dict = {
        'observedValue':str(col_value),
        'permissibleValue':{
            'value': output_value,
            'conceptCode': None
        }
    }
    return output_dict


def score_value_match(value_dict_list,cde_index,g):
    query = "MATCH (n:CDE) WHERE ID(n) = {0:d} RETURN n.VALUE_DOMAIN_TYPE".format(int(cde_index))
    result = query_graph(query,g)
    value_domain_type = result.value()[0]
    if value_domain_type == "Enumerated":
        correct = [i for i in value_dict_list if i['permissibleValue']['value'] != "NOMATCH"]
        points = len(correct)
        pct = points/len(value_dict_list)
    else:
        correct = [i for i in value_dict_list if i['permissibleValue']['value'] == "CONFORMING"]
        points = len(correct)
        pct = points/len(value_dict_list)
    return pct



def classify_column(column,g):
    """
    Determine which Common Data Elements (CDEs) in a graph database
    are best matches to an input column.

    Parameters:
    column (pandas.Series): Column from input dataframe.
    graph (neo4j.DirectDriver): Connection to neo4j database representation
        of Thesaurus and caDSR data in prescribed format.

    Returns:
    List of ranked results by CDE_ID
    """
