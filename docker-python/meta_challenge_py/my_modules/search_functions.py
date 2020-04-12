 #!/usr/bin/python3

import pandas as pd
import numpy as np
from my_modules import utils


X_FT_STRUCTURE = {
    'index':{
        'column_no': 0,
        'ft_postprocess_params': {}
    },
    'cde_id':{
        'column_no': 1,
        'ft_postprocess_params': {}
    },
    'syn_classsum':{
        'column_no': 2,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_CLASS] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'sum'
            }
        }
    },
    'syn_propsum':{
        'column_no': 3,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_PROP] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'sum'
            }
        }
    },
    'syn_objsum':{
        'column_no': 4,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_OBJ] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'sum'
            }
        }
    },
    'syn_classmax':{
        'column_no': 5,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_CLASS] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'max'
            }
        }
    },
    'syn_propmax':{
        'column_no': 6,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_PROP] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'max'
            }
        }
    },
    'syn_objmax':{
        'column_no': 7,
        'ft_postprocess_params': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_OBJ] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'max'
            }
        }
    },
    'ftsearch_cde':{
        'column_no': 8,
        'ft_postprocess_params': {
            'CDE_Name':{
                'query': lambda z: f"MATCH (n) - [:IS_SHORT] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'max'
            },
            'CDE':{
                'query': lambda z: f"MATCH (n:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n) AS node_id, ID(n), n.CDE_ID",
                'aggregation': 'max'
            }
        }
    },
    'ftsearch_dec':{
        'column_no': 9,
        'ft_postprocess_params': {
            'DEC':{
                'query': lambda z: f"MATCH (n) - [:IS_CAT] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'sum'
            }
        }
    },
    'ftsearch_question':{
        'column_no': 10,
        'ft_postprocess_params': {
            'QuestionText':{
                'query': lambda z: f"MATCH (n) - [:QUESTION] - (m:CDE) WHERE ID(n) IN [{z}] RETURN DISTINCT ID(n), ID(m), m.CDE_ID",
                'aggregation': 'max'
            }
        }
    }
}




def find_synonyms(input_str,g):
    query = "MATCH (n:Synonym) WHERE \"{0:s}\" CONTAINS n.name_lower RETURN n.name_lower".format(str(input_str).lower())
    result = utils.query_graph(query,g)
    values = result.value()
    return values


def full_str_match_synonym(col_name,g):
    string_list = expand_string_names(col_name)
    where_clause_list = ["n.lower = '" + i + "'" for i in string_list]
    where_clause = " OR ".join(where_clause_list)
    query = "MATCH (n:Synonym) WHERE " + where_clause + "RETURN n.name"
    result = utils.query_graph(query,g)
    return result.value()

def full_str_match_short(col_name,g):
    string_list = expand_string_names(col_name)
    where_clause_list = ["n.cde_short_name_lower = '" + i + "'" for i in string_list]
    where_clause = " OR ".join(where_clause_list)
    query = "MATCH (n:CDE) WHERE " + where_clause + "RETURN n.name"
    result = utils.query_graph(query,g)
    return result.value()

def nameindex_query(input_string,g):
    query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node, score RETURN ID(node), score, LABELS(node)".format(str(input_string))
    result = utils.query_graph(query,g)
    return result.values()

def nameindex_query_multiple(input_string_list,g,score_coef=1, MIN_SCORE = 0):
    if isinstance(score_coef,list):
        if len(score_coef) != len(input_string_list):
            raise ValueError("nameindex query score coef length mismatch")
        else:
            score_coef_list = score_coef
    else:
        score_coef_list = [score_coef] * len(input_string_list)
    query = "CALL {\n"
    input_string = input_string_list[0]
    query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node, score \n".format(str(input_string))
    query += " WHERE score > {0:1.2f}\n".format(MIN_SCORE/score_coef_list[0])
    query += " RETURN ID(node) as node_id, {0:1.2f} * score AS normal_score, LABELS(node) as node_labels\n".format(score_coef_list[0])
    for i in range(1,len(input_string_list)):
        input_string = input_string_list[i]
        query += " UNION ALL"
        query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node, score \n".format(str(input_string))
        query += " WHERE score > {0:1.2f}\n".format(MIN_SCORE/score_coef_list[i])
        query += " RETURN ID(node) as node_id, {0:1.2f} * score AS normal_score, LABELS(node) as node_labels\n".format(score_coef_list[i])
    query += "}\n"
    query += "RETURN node_id,MAX(normal_score),node_labels\n"
    result = utils.query_graph(query,g)
    return result.values()



def enumeration_exact_search(unique_values_list,g):
    unique_values_lower = ["'" + str(i).lower() + "'" for i in unique_values_list]
    unique_values_lower = list(set(unique_values_lower))
    query = "MATCH (n:AnswerText) - [:CAN_BE] - (m:Answer) - [:PERMITS] - (c:CDE) WHERE n.name_lower in [{0:s}] ".format(",".join(unique_values_lower))
    query += "WITH DISTINCT n.name_lower AS name_lower, c AS c_distinct "
    query += "RETURN ID(c_distinct), c_distinct.CDE_ID, COUNT(*)*1.0/{0:d}".format(len(unique_values_list))
    result = utils.query_graph(query,g)
    values = result.values()
    values.sort(key = lambda z: z[2], reverse=True)
    return values


def enumeration_ansindex_single_search(single_unique_value,g):
    query = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
    query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer) - [:PERMITS] - (c:CDE) \n"
    query += " RETURN ID(c), c.CDE_ID, MAX(score)"
    result = utils.query_graph(query,g)
    return result.values()


def enumeration_concept_single_search(single_unique_value,g):
    query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(single_unique_value))
    query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
    query += " RETURN ID(c), c.CDE_ID, MAX(score)"
    result = utils.query_graph(query,g)
    values = result.values()
    values.sort(key = lambda z: z[2], reverse=True)
    return values


def enumeration_concept_search(value_list,g):
    value_set = list(set(value_list))
    query = "CALL {\n"
    v = value_set[0]
    query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
    query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
    query += " RETURN ID(c) AS cde_index, c.CDE_ID as cde_id, MAX(score) AS max_score \n"
    for v in value_set[1:len(value_set)]:
        query += " UNION ALL\n"
        query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
        query += " MATCH (n:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
        query += " RETURN ID(c) AS cde_index, c.CDE_ID as cde_id, MAX(score) AS max_score \n"
    query += "}\n"
    query += "RETURN cde_index, cde_id, SUM(max_score) * 1.0 / {0:d}".format(len(value_set))
    result = utils.query_graph(query,g)
    values = result.values()
    if len(values) > 1:
        values.sort(key = lambda z: z[2], reverse=True)
    return values

def enumeration_ansindex_search(value_list,g):
    value_set = list(set(value_list))
    query = "CALL {\n"
    v = value_set[0]
    query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
    query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
    query += " RETURN ID(c) AS cde_index, c.CDE_ID as cde_id, MAX(score) AS max_score \n"
    for v in value_set[1:len(value_list)]:
        query += " UNION ALL\n"
        query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node AS n, score\n".format(str(v))
        query += " MATCH (n:AnswerText) - [:CAN_BE] - (a:Answer) - [:PERMITS] - (c:CDE)\n"
        query += " RETURN ID(c) AS cde_index, c.CDE_ID as cde_id, MAX(score) AS max_score \n"
    query += "}\n"
    query += "RETURN cde_index, cde_id, SUM(max_score) * 1.0 / {0:d}".format(len(value_set))
    result = utils.query_graph(query,g)
    values = result.values()
    if len(values) > 1:
        values.sort(key = lambda z: z[2], reverse=True)
    return values

def get_enum_answers(cde_index,g):
    query = "MATCH (n:CDE) - [:PERMITS] - (a:Answer) - [:CAN_BE] - (at: AnswerText) WHERE ID(n) = {0:d} RETURN ID(at),at.name".format(cde_index)
    result = utils.query_graph(query,g)
    values = result.values()
    return values

def get_CDEs(ft_result,update_column,g):
    node_type = ft_result[2][0]
    node_index = ft_result[0]
    query = X_FT_STRUCTURE[update_column]['ft_postprocess_params'][node_type]['query'](node_index)
    result = utils.query_graph(query,g)
    values = result.values()
    return values

def create_data_row(input_row):
    cols = [(i,X_FT_STRUCTURE[i]['column_no']) for i in X_FT_STRUCTURE]
    cols.sort(key=lambda z: z[1])
    d = {}
    for c in cols:
        if c[0] == 'index':
            d[c[0]] = [input_row[0]]
        elif c[0] == 'cde_id':
            d[c[0]] = [input_row[1]]
        else:
            d[c[0]] = [0]
    return pd.DataFrame(d)

def update_data(df,ft_result,update_column,update_cde_index,g):
    plus_value = ft_result[1]
    node_type = ft_result[2][0]
    if update_column not in df.columns:
        raise ValueError("{0:s} column not in DataFrame".format(update_column))
    new_value = X_FT_STRUCTURE[update_column]['ft_postprocess_params'][node_type]['aggregation'](
        df[update_column].loc[df['index']==update_cde_index].values[0],
        plus_value
    )
    df.loc[df['index']==update_cde_index,update_column] = new_value
    return df


def create_or_update(df,ft_result,update_column,g):
    if update_column not in df.columns:
        raise ValueError("{0:s} column not in DataFrame".format(update_column))
    update_cdes = get_CDEs(ft_result,update_column,g)
    for update_row in update_cdes:
        update_index = update_row[0]
        if update_index not in df['index'].values:
            new_row = create_data_row(update_row)
            df = df.append(new_row)
        df = update_data(df,ft_result,update_column,update_index,g)
    return df


def score_enumeration_check(column_series):
    no_nan = column_series.loc[column_series == column_series]
    l_all = len(no_nan)
    l = len(no_nan.unique())
    if (l < (np.sqrt(l_all))) and (str(no_nan.dtype)[0] not in ['f','i','u']):
        return True
    else:
        return False


def score_synonym_str(str1_full,str2_wordsonly,input_str):
    pct_coverage = len(str2_wordsonly.replace(" ",""))/len(input_str)
    space_count = str1_full.count(" ")
    out_word_count = space_count - str2_wordsonly.count(" ")
    word_lengths = [len(i) for i in str1_full.split(" ")]
    out_word_lengths = [len(i) for i in str1_full.split(" ") if i not in str2_wordsonly.split(" ")]
    score = pct_coverage * (1-(space_count+1)/(len(input_str)))
    return score


def nans_vs_nexp(n_ans,n_exp):
    if n_exp == 0:
        return 0
    else:
        return 1-(np.abs(n_ans-n_exp)/(n_ans + n_exp))


def create_answer_count_df(cde_indices,g):
    q = "MATCH (n:CDE) - [:PERMITS] - (m:Answer) WHERE ID(n) IN [{0:s}] RETURN ID(n),COUNT(*)".format(",".join(cde_indices.astype('int').astype('str')))
    result = utils.query_graph(q,g)
    answer_counts = result.values()
    q2 = "MATCH (n:CDE) WHERE ID(n) IN [{0:s}] AND n.DATATYPE = 'BOOLEAN' RETURN ID(n),2".format(",".join(cde_indices.astype('int').astype('str')))
    result = utils.query_graph(q2,g)
    answer_counts2 = result.values()
    n1 = [i[0] for i in answer_counts]
    answer_counts2 = [i for i in answer_counts2 if i[0] not in n1]
    answer_count_df = pd.DataFrame(answer_counts + answer_counts2,columns=['index','answer_count'])
    return answer_count_df


def create_new_search_strings(input_str,g,min_substr_length):
    word_list = find_synonyms(input_str,g)
    word_list = [w for w in word_list if len(w) >= min_substr_length]
    exclusive_groups = utils.find_exclusive_groups(input_str,word_list)
    strings = list(set([utils.create_string(i,input_str,word_list) for i in exclusive_groups]))
    scores = [score_synonym_str(i[0],i[1],input_str) for i in strings]
    combined_list = [[strings[i],scores[i]] for i in range(len(strings))]
    combined_list.sort(key=lambda z: z[1], reverse=True)
    return [(i[0][0],i[1]) for i in combined_list[0:3] if i[1] > 0]

def score_enum_values(unique_values,cde_indices,g, MIN_SCORE = 0):
    query = "CALL {\n"
    query += " MATCH (c:CDE) WHERE ID(c) IN [{0:s}] \n".format(",".join([str(i) for i in cde_indices]))
    query += " RETURN ID(c) as cde_index, 0 AS g_max \n"
    for u in unique_values:
        query += " UNION ALL "
        query += " CALL {\n"
        query += " CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node as a, score ".format(str(u))
        query += " MATCH (a:AnswerText) - [:CAN_BE] - (ans:Answer) - [:PERMITS] - (c:CDE)\n "
        # query += " WHERE ID(c) IN [{0:s}] \n".format(",".join([str(i) for i in cde_indices]))
        query += " RETURN ID(c) as cde_index, CASE MAX(score) > {0:1.2f} WHEN TRUE THEN 1 ELSE 0 END as g".format(float(MIN_SCORE))
        query += " UNION ALL \n"
        query += " CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node as s, score ".format(str(u))
        query += " MATCH (s:Synonym) - [:IS_CALLED] - (con:Concept) - [:EQUALS] - (ans:Answer) - [:PERMITS] - (c:CDE)\n "
        # query += " WHERE ID(c) IN [{0:s}] \n".format(",".join([str(i) for i in cde_indices]))
        query += " RETURN ID(c) as cde_index, CASE MAX(score) > {0:1.2f} WHEN TRUE THEN 1 ELSE 0 END as g".format(float(MIN_SCORE))
        query += " }\n"
        query += " RETURN cde_index, MAX(g) AS g_max \n"
    query += "}\n"
    query += "RETURN cde_index,SUM(g_max) * 1.0 / {0:d}".format(len(unique_values))
    result = utils.query_graph(query,g)
    values = result.values()
    values.sort(key=lambda z: z[1], reverse=True)
    return values


def score_value_match(value_dict_list,cde_index,g):
    query = "MATCH (n:CDE) WHERE ID(n) = {0:d} RETURN n.VALUE_DOMAIN_TYPE".format(int(cde_index))
    result = utils.query_graph(query,g)
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

