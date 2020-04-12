import pandas as pd
from my_modules import utils,search_functions,score_functions,datachecks
import numpy as np


DISPLAY_FORMATS = [
    "mm/dd/yy",
    "MM/DD/YYYY",
    "DD/MON/YYYY",
    "YYYY-MM-DD",
    "YYYY",
    "TIME (HR(24):MN)",
    "YYYYMMDD",
    "9999.99",
    "mm/dd/yyyy",
    "9999999",
    "10,3",
    "9999.9",
    "%",
    "999.9",
    "99.9",
    "hh:mm:ss",
    "999.99",
    "9.999",
    "999999.9",
    "hh:mm",
    "hh:mm:ss:rr",
    "TIME_MIN",
    "99.99",
    "9999.999",
    "MMYYYY",
    "TIME_HH:MM",
    "99999.99",
    "MMDDYYYY",
    "999-99-9999",
    "YYYYMM"
]

DATATYPES = [
    "Integer",
    "CHARACTER",
    "DATETIME",
    "SAS Date",
    "SAS Time",
    "NUMBER",
    "varchar",
    "DATE",
    "ALPHANUMERIC",
    "TIME",
    "HL7EDv3",
    "BOOLEAN",
    "binary",
    "Numeric Alpha DVG",
    "Date Alpha DVG",
    "Derived",
    "UMLUidv1.0",
    "DATE/TIME",
    "HL7STv3",
    "CLOB",
    "HL7CDv3",
    "HL7INTv3",
    "HL7REALv3",
    "HL7TSv3",
    "HL7PNv3",
    "HL7TELv3",
    "OBJECT",
    "Alpha DVG"
]
col_series = df[df.columns[85]]
g = graphDB
NAMEINDEX_SEARCH_REQD = 25
NAMEINDEX_CDE_REQD = 5
MIN_SCORE = 0
FOLLOW_ON_SEARCH_MIN_WORD_LEN = 3

if 1:
    import time
    t = time.time()
    col_name = col_series.name
    search_string = utils.clean_string_for_fulltext(
        utils.lower_upper_split(
            utils.period_replace(
                utils.underscore_replace(
                    col_name
                )
            )
        )
    )
    df = pd.DataFrame(columns=['index','cde_id'])
    result_types = list(set([j for i in search_functions.X_FT_STRUCTURE for j in search_functions.X_FT_STRUCTURE[i]['ft_postprocess_params']]))
    result_type_dict = {i:[j for j in search_functions.X_FT_STRUCTURE if i in search_functions.X_FT_STRUCTURE[j]['ft_postprocess_params']] for i in result_types}
    search_results = search_functions.nameindex_query_multiple([search_string],g)
    print("Initial search complete, {0:d} results".format(len(search_results)))
    print("{0:1.2f} seconds.\n".format(time.time()-t))
    if (len(search_results) < NAMEINDEX_SEARCH_REQD) or (len([i for i in search_results if i[2][0] in ['CDE','CDE_Name','DEC','QuestionText']]) < NAMEINDEX_CDE_REQD):
        min_substr_length = max(1/2 * np.sqrt(len(search_string)),FOLLOW_ON_SEARCH_MIN_WORD_LEN)
        new_search_strings = [(search_string,1)] + search_functions.create_new_search_strings(search_string,g,min_substr_length)
        print("New Search Strings created, {0:d} strings".format(len(new_search_strings)))
        print("{0:1.2f} seconds.\n".format(time.time()-t))
        search_results = search_functions.nameindex_query_multiple([utils.clean_string_for_fulltext(s[0]) for s in new_search_strings],g,[s[1] for s in new_search_strings])
        print("Follow-on search complete, {0:d} results".format(len(search_results)))
        print("{0:1.2f} seconds.\n".format(time.time()-t))
    for col in search_functions.X_FT_STRUCTURE:
        search_score_df = pd.DataFrame(columns=['index','cde_id',col])
        for result_type in search_functions.X_FT_STRUCTURE[col]['ft_postprocess_params']:
            search_result_filtered = [(i[0],i[1]) for i in search_results if i[2][0] == result_type]
            search_result_df = pd.DataFrame(search_result_filtered,columns=['node_index',col])
            q = search_functions.X_FT_STRUCTURE[col]['ft_postprocess_params'][result_type]['query'](",".join([str(node_index_int) for node_index_int in search_result_df['node_index'].tolist()]))
            agg_type = search_functions.X_FT_STRUCTURE[col]['ft_postprocess_params'][result_type]['aggregation']
            res = utils.query_graph(q,g)
            res_df = pd.DataFrame(res.values(),columns = ['node_index','index','cde_id'])
            scored_results = pd.merge(res_df,search_result_df,how='left',on='node_index')
            if agg_type == 'max':
                agg_results = scored_results[['index','cde_id',col]].groupby(by=['index','cde_id'],axis=0,as_index=False).max()
            elif agg_type == 'sum':
                agg_results = scored_results[['index','cde_id',col]].groupby(by=['index','cde_id'],axis=0,as_index=False).sum()
            search_score_df = pd.concat([search_score_df,agg_results])
        if search_score_df.shape[0] > 0:
            final_agg = search_score_df.groupby(by=['index','cde_id'],axis=0,as_index=False).max()
            df = pd.merge(df,final_agg,on=['index','cde_id'],how='outer')
        elif col not in df.columns.tolist():
            df[col] = 0
