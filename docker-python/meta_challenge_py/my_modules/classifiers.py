#!/usr/bin/python3

import pandas as pd
from my_modules.utils import *

X_STRUCTURE = {
    'index':{
        'column_no': 0,
        'score_functions': None
    },
    'ftsearch_syn_class':{
        'column_no': 1,
        'score_functions': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_CLASS] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_syn_prop':{
        'column_no': 2,
        'score_functions': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_PROP] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_syn_obj':{
        'column_no': 3,
        'score_functions': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_OBJ] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_cde':{
        'column_no': 4,
        'score_functions': {
            'CDE_Name':{
                'query': lambda z: f"MATCH (n) - [:ISSHORT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
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
        'score_functions': {
            'DEC':{
                'query': lambda z: f"MATCH (n) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'ftsearch_question':{
        'column_no': 6,
        'score_functions': {
            'QuestionText':{
                'query': lambda z: f"MATCH (n) - [:QUESTION] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: max([old,new])
            }
        }
    },
    'ans_match':{
        'column_no': 10,
        'score_functions': {

        }
    },
    'ans_enum':{
        'column_no': 11,
        'score_functions': {

        }
    },
    'synsearch_class':{
        'column_no': 12,
        'score_functions': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_CLASS] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'synsearch_prop':{
        'column_no': 13,
        'score_functions': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_PROP] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    },
    'synsearch_obj':{
        'column_no': 14,
        'score_functions': {
            'Synonym':{
                'query': lambda z: f"MATCH (n) - [:IS_CALLED] - (:Concept) - [:IS_OBJ] - (:DEC) - [:IS_CAT] - (m:CDE) WHERE ID(n) = {z} RETURN ID(m)",
                'aggregation': lambda old,new: old + new
            }
        }
    }
}

DATA_TYPES = {
    'Integer':{
        'type': 'int'
    },
    'CHARACTER':{
        'type': 'str'
    },
    'DATETIME':{
        'type': 'str'
    },
    'SAS Date':{
        'type': 'str'
    },
    'SAS Time':{
        'type': 'str'
    },
    'NUMBER':{
        'type': 'num'
    },
    'varchar':{
        'type': 'str'
    },
    'DATE':{
        'type': 'str'
    },
    'ALPHANUMERIC':{
        'type': 'str'
    },
    'TIME':{
        'type': 'str'
    },
    'HL7EDv3':{
        'type': 'str'
    },
    'BOOLEAN':{
        'type': 'str'
    },
    'binary':{
        'type': 'bit'
    },
    'Numeric Alpha DVG':{
        'type': 'str'
    },
    'Date Alpha DVG':{
        'type': 'str'
    },
    'Derived':{
        'type': 'str'
    },
    'UMLUidv1.0':{
        'type': 'str'
    },
    'DATE/TIME':{
        'type': 'str'
    },
    'HL7STv3':{
        'type': 'str'
    },
    'CLOB':{
        'type': 'str'
    },
    'HL7CDv3':{
        'type': 'str'
    },
    'HL7INTv3':{
        'type': 'int'
    },
    'HL7REALv3':{
        'type': 'num'
    },
    'HL7TSv3':{
        'type': 'str'
    },
    'HL7PNv3':{
        'type': 'str'
    },
    'HL7TELv3':{
        'type': 'str'
    },
    'OBJECT':{
        'type': 'str'
    },
    'Alpha DVG':{
        'type': 'str'
    }
}

def find_words(col_name,synonyms_df):
    found_boolean = synonyms_df['name_lower'].apply(lambda z: z in col_name.lower())
    all_words = synonyms_df.loc[found_boolean]
    lengths = [(i,len(all_words['name_lower'].loc[i])) for i in all_words.index]
    lengths.sort(key=lambda z: z[1],reverse=True)
    words = [i[0] for i in lengths]
    return words


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
    cols = [(i,X_STRUCTURE[i]['column_no']) for i in X_STRUCTURE]
    cols.sort(key=lambda z: z[1])
    d = {i[0]:[input_index] if i[0]=='index' else [0] for i in cols}
    return pd.DataFrame(d)

def update_data(df,ft_result,update_column):
    plus_value = ft_result[1]
    node_type = ft_result[2][0]
    if update_column not in df.columns:
        raise ValueError("{0:s} column not in DataFrame",update_column)
    new_value = X_STRUCTURE[update_column]['score_functions'][node_type]['aggregation'](
        df[update_column].loc[df['index']==update_index],
        plus_value
    )
    df[update_column].loc[df['index']==update_index] = new_value


def create_or_add(df,ft_result,update_column):
    if update_column not in df.columns:
        raise ValueError("{0:s} column not in DataFrame",update_column)
    if update_index not in df['index'].values:
        new_row = create_data_row(update_index)
        df = df.append(new_row)
    add_data(df,ft_result,update_column)
    return df

def enumeration_exact_search(unique_values_list,g):
    unique_values_lower = ["'" + str(i).lower() + "'" for i in unique_values_list]
    unique_values_lower = list(set(unique_values_lower))
    query = "MATCH (n:AnswerText) - [:CANBE] - (m:Answer) - [:PERMITS] - (c:CDE) WHERE n.name_lower IN [" + ",".join(unique_values_lower) + "] RETURN n.name_lower,ID(m),c.CDE_ID"
    result = query_graph(query,g)
    values = result.values()
    cdes = [i[2] for i in values]
    unique_cdes = list(set(cdes))
    cde_scores = [(u,cdes.count(u)/len(unique_values_lower)) for u in unique_cdes]
    return values
    


def get_score_indices(
        ft_result,
        update_column,
        g
    ):
    if update_column not in X_STRUCTURE:
        raise ValueError("{0:s} column not in DataFrame",update_column)
    if ft_result[2][0] in X_STRUCTURE[update_column]['score_functions']:
        query = X_STRUCTURE[update_column]['score_functions'][ft_result[2][0]]['query'](ft_result[0])
        result = query_graph(query,g)
        values = result.value()
        return values
    else:
        return None

def get_enum_answers(cde_index,g):
    query = "MATCH (n:CDE) - [:PERMITS] - (a:Answer) - [:CANBE] - (at: AnswerText) WHERE ID(n) = {0:d} RETURN ID(at),at.name".format(cde_index)
    result = query_graph(query,g)
    values = result.values()
    return values


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
