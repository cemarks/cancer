import pandas as pd
from my_modules import utils,search_functions,score_functions,datachecks
import numpy as np

NO_VAR_VALUE = 0

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
    "Alpha DVG",
    "xsd:string",
    "xsd:boolean",
    "UMLBinaryv1.0",
    "UMLUriv1.0",
    "UMLOctetv1.0",
    "UMLCodev1.0",
    "UMLXMLv1.0",
    "xsd:dateTime"
]



def build_initial_column_data(
    col_series,
    g, 
    NAMEINDEX_SEARCH_REQD = 25,
    NAMEINDEX_CDE_REQD = 5,
    MIN_SCORE = 0,
    FOLLOW_ON_SEARCH_MIN_WORD_LEN = 3,
    PRINT_STATS = False
):
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
    if PRINT_STATS:
        print("Initial search complete, {0:d} results".format(len(search_results)))
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
            else:
                agg_results = pd.DataFrame(columns=['index','cde_id',col])
            search_score_df = pd.concat([search_score_df,agg_results])
        if search_score_df.shape[0] > 0:
            final_agg = search_score_df.groupby(by=['index','cde_id'],axis=0,as_index=False).max()
            df = pd.merge(df,final_agg,on=['index','cde_id'],how='outer')
        elif col not in df.columns.tolist():
            df[col] = 0
    df['secondary_search'] = [0] * df.shape[0]
    for col in search_functions.X_FT_STRUCTURE:
        df.loc[df[col] != df[col],col] = 0
    ordered_cols = [(col,search_functions.X_FT_STRUCTURE[col]['column_no']) for col in search_functions.X_FT_STRUCTURE]
    ordered_cols.sort(key = lambda z: z[1])
    df = df[[o[0] for o in ordered_cols]]
    if (len(search_results) < NAMEINDEX_SEARCH_REQD) or (len([i for i in search_results if i[2][0] in ['CDE','CDE_Name','DEC','QuestionText']]) < NAMEINDEX_CDE_REQD):
        new_df = pd.DataFrame(columns=['index','cde_id'])
        min_substr_length = max(int(np.floor(np.sqrt(len(search_string)* 1./2))),FOLLOW_ON_SEARCH_MIN_WORD_LEN)
        new_search_strings = [(search_string,1)] + search_functions.create_new_search_strings(search_string,g,min_substr_length)
        if PRINT_STATS:
            print("New Search Strings created, {0:d} strings".format(len(new_search_strings)))
            print("{0:1.2f} seconds.\n".format(time.time()-t))
        new_search_results = search_functions.nameindex_query_multiple([utils.clean_string_for_fulltext(s[0]) for s in new_search_strings],g,[s[1] for s in new_search_strings])
        if PRINT_STATS:
            print("Follow-on search complete, {0:d} results".format(len(new_search_results)))
            print("{0:1.2f} seconds.\n".format(time.time()-t))
        for col in search_functions.X_FT_STRUCTURE:
            search_score_df = pd.DataFrame(columns=['index','cde_id',col])
            for result_type in search_functions.X_FT_STRUCTURE[col]['ft_postprocess_params']:
                search_result_filtered = [(i[0],i[1]) for i in new_search_results if i[2][0] == result_type]
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
                else:
                    agg_results = pd.DataFrame(columns=['index','cde_id',col])
                search_score_df = pd.concat([search_score_df,agg_results])
            if search_score_df.shape[0] > 0:
                final_agg = search_score_df.groupby(by=['index','cde_id'],axis=0,as_index=False).max()
                new_df = pd.merge(new_df,final_agg,on=['index','cde_id'],how='outer')
            elif col not in new_df.columns.tolist():
                    new_df[col] = 0
        new_df['secondary_search'] = [1] * new_df.shape[0]
        for col in search_functions.X_FT_STRUCTURE:
            new_df.loc[new_df[col] != new_df[col],col] = 0
        ordered_cols = [(col,search_functions.X_FT_STRUCTURE[col]['column_no']) for col in search_functions.X_FT_STRUCTURE]
        ordered_cols.sort(key = lambda z: z[1])
        new_df = new_df[[o[0] for o in ordered_cols]]
        new_df = new_df[~new_df['index'].isin(df['index'])]
        df = pd.concat([df,new_df])
    if PRINT_STATS:
        print("Initial df created, {0:d} rows".format(df.shape[0]))
        print("{0:1.2f} seconds.\n".format(time.time()-t))
    unique_values = utils.col_unique_values(col_series)
    if len(unique_values) > 0:
        unique_values_clean = [utils.clean_string_for_fulltext(i) for i in unique_values]
        enum_search1 = search_functions.enumeration_concept_search(unique_values_clean,g)
        enum_search1_df = pd.DataFrame([es for es in enum_search1 if es[2] > MIN_SCORE],columns = ['index','cde_id','enum_concept_search'])
        enum_search2 = search_functions.enumeration_ansindex_search(unique_values_clean,g)
        enum_search2_df = pd.DataFrame([es for es in enum_search2 if es[2] > MIN_SCORE],columns = ['index','cde_id','enum_answer_search'])
        df = pd.merge(
            df,
            enum_search1_df,
            on = ['index','cde_id'],
            how = 'outer'
        )
        df = pd.merge(
            df,
            enum_search2_df,
            on = ['index','cde_id'],
            how = 'outer'
        )
    else:
        df['enum_concept_search'] = 0
        df['enum_answer_search'] = 0
    if PRINT_STATS:
        print("Enum searches complete, {0:d} rows".format(df.shape[0]))
        print("{0:1.2f} seconds.\n".format(time.time()-t))
    if df.shape[0] > 0:
        answer_count_df = search_functions.create_answer_count_df(df['index'].values,g)
        answer_count_df = pd.merge(df['index'],answer_count_df, on='index',how='outer')
        # print("Answer Count DF Created: {0:d} sec".format(int(time.time() - t)))
        # print("Answer Count DF size: {0:d}.\n".format(answer_count_df.shape[0]))
        n_ans = len(unique_values)
        n_lines = sum(col_series==col_series)
        answer_count_df.loc[answer_count_df['answer_count'] != answer_count_df['answer_count'],'answer_count'] = n_lines
        if answer_count_df.shape[0] > 0:
            answer_count_df = pd.DataFrame(
                {
                    'index': answer_count_df['index'],
                    'answer_count_score': answer_count_df.apply(lambda z: search_functions.nans_vs_nexp(n_ans,z[1]), axis=1)
                }
            )
        else:
            answer_count_df = pd.DataFrame(
                {
                    'index': [],
                    'answer_count_score': []
                }
            )
        df = pd.merge(df,answer_count_df,on='index',how='inner')
        if PRINT_STATS:
            print("Answer count complete, {0:d} rows".format(df.shape[0]))
            print("{0:1.2f} seconds.\n".format(time.time()-t))
        for c in df.columns:
            v = df[c] != df[c]
            if any(v):
                df.loc[v,c] = 0
        if n_ans > 0:
            query = "MATCH (n:CDE) WHERE ID(n) IN [{0:s}] RETURN DISTINCT ID(n), n.DATATYPE, n.DISPLAY_FORMAT, n.VALUE_DOMAIN_TYPE".format(",".join([str(i) for i in df['index'].values]))
            result = utils.query_graph(query,g)
            values = result.values()
            temp_df = pd.DataFrame(values,columns = ['index','datatype','display_format','value_domain_type'])
            enum_ids = list(temp_df['index'].loc[temp_df['value_domain_type']=='Enumerated'].values)
            enum_scores = search_functions.score_enum_values(unique_values_clean,enum_ids,g)
            enum_score_df = pd.DataFrame(enum_scores,columns = ['index','value_score'])
            temp_df = pd.merge(temp_df,enum_score_df,on='index',how='left')
            for display_format in DISPLAY_FORMATS:
                temp_df.loc[(temp_df['value_domain_type']=='NonEnumerated') & (temp_df['display_format'] == display_format),'value_score'] = len([j for j in unique_values if datachecks.check_display_format(str(j),display_format)])/len(unique_values)
            for datatype in DATATYPES:
                temp_df.loc[(temp_df['value_domain_type']=='NonEnumerated') & (temp_df['display_format'].isnull()) & (temp_df['datatype'] == datatype),'value_score'] = len([j for j in unique_values if datachecks.check_datatype(str(j),datatype)])/len(unique_values)
            df = pd.merge(
                df,
                temp_df[['index','value_score']],
                on = 'index',
                how = 'inner'
            )
        else:
            df['value_score'] = 0
        if PRINT_STATS:
            print("Enum scores complete, {0:d} rows".format(df.shape[0]))
            print("{0:1.2f} seconds.\n".format(time.time()-t))
        df['index'] = df['index'].astype('int')
    for col in df.columns:
        df.loc[df[col] != df[col],col] = 0
    return df


def expand_column_X(
    column_small_X
):
    df = column_small_X.copy()
    if df.shape[0] > 0:
        max_cde = max(column_small_X['ftsearch_cde'])
        max_dec = max(column_small_X['ftsearch_dec'])
        max_que = max(column_small_X['ftsearch_question'])
        max_syn_classsum = max(column_small_X['syn_classsum'])
        max_syn_propsum = max(column_small_X['syn_propsum'])
        max_syn_objsum = max(column_small_X['syn_objsum'])
        max_syn_classmax = max(column_small_X['syn_classmax'])
        max_syn_propmax = max(column_small_X['syn_propmax'])
        max_syn_objmax = max(column_small_X['syn_objmax'])
        max_enum_concept = max(column_small_X['enum_concept_search'])
        max_enum_ans = max(column_small_X['enum_answer_search'])
        max_ans_score = max(column_small_X['answer_count_score'])
        max_val_score = max(column_small_X['value_score'])
        mean_cde = np.mean(column_small_X['ftsearch_cde'])
        mean_dec = np.mean(column_small_X['ftsearch_dec'])
        mean_question = np.mean(column_small_X['ftsearch_question'])
        mean_syn_classsum = np.mean(column_small_X['syn_classsum'])
        mean_syn_propsum = np.mean(column_small_X['syn_propsum'])
        mean_syn_objsum = np.mean(column_small_X['syn_objsum'])
        mean_syn_classmax = np.mean(column_small_X['syn_classmax'])
        mean_syn_propmax = np.mean(column_small_X['syn_propmax'])
        mean_syn_objmax = np.mean(column_small_X['syn_objmax'])
        mean_enum_concept = np.mean(column_small_X['enum_concept_search'])
        mean_enum_ans = np.mean(column_small_X['enum_answer_search'])
        mean_ans_score = np.mean(column_small_X['answer_count_score'])
        mean_val_score = np.mean(column_small_X['value_score'])
        std_cde = np.std(column_small_X['ftsearch_cde'])
        std_dec = np.std(column_small_X['ftsearch_dec'])
        std_question = np.std(column_small_X['ftsearch_question'])
        std_syn_classsum = np.std(column_small_X['syn_classsum'])
        std_syn_propsum = np.std(column_small_X['syn_propsum'])
        std_syn_objsum = np.std(column_small_X['syn_objsum'])
        std_syn_classmax = np.std(column_small_X['syn_classmax'])
        std_syn_propmax = np.std(column_small_X['syn_propmax'])
        std_syn_objmax = np.std(column_small_X['syn_objmax'])
        std_enum_concept = np.std(column_small_X['enum_concept_search'])
        std_enum_ans = np.std(column_small_X['enum_answer_search'])
        std_ans_score = np.std(column_small_X['answer_count_score'])
        std_val_score = np.std(column_small_X['value_score'])
        max_secondary_search = np.mean(column_small_X['secondary_search'])
        n = column_small_X.shape[0]
        df['max_cde'] = max_cde
        df['max_dec'] = max_dec
        df['max_que'] = max_que
        df['max_syn_classsum'] = max_syn_classsum
        df['max_syn_propsum'] = max_syn_propsum
        df['max_syn_objsum'] = max_syn_objsum
        df['max_syn_classmax'] = max_syn_classmax
        df['max_syn_propmax'] = max_syn_propmax
        df['max_syn_objmax'] = max_syn_objmax
        df['max_enum_concept'] = max_enum_concept
        df['max_enum_ans'] = max_enum_ans
        df['max_ans_score'] = max_ans_score
        df['max_val_score'] = max_val_score
        df['max_secondary_search'] = max_secondary_search
        df['n'] = n
        df['pct_cde'] = sum(df['ftsearch_cde'] > 0)/n
        df['pct_dec'] = sum(df['ftsearch_dec'] > 0)/n
        df['pct_que'] = sum(df['ftsearch_question'] > 0)/n
        df['pct_syn_classsum'] = sum(df['syn_classsum'] > 0)/n
        df['pct_syn_propsum'] = sum(df['syn_propsum'] > 0)/n
        df['pct_syn_objsum'] = sum(df['syn_objsum'] > 0)/n
        df['pct_syn_classmax'] = sum(df['syn_classmax'] > 0)/n
        df['pct_syn_propmax'] = sum(df['syn_propmax'] > 0)/n
        df['pct_syn_objmax'] = sum(df['syn_objmax'] > 0)/n
        df['pct_enum_concept'] = sum(df['enum_concept_search'] > 0)/n
        df['pct_enum_ans'] = sum(df['enum_answer_search'] > 0)/n
        df['pct_ans_score'] = sum(df['answer_count_score'] > 0)/n
        df['pct_val_score'] = sum(df['value_score'] > 0)/n
        df['pct_secondary_search'] = sum(df['secondary_search'] > 0)/n
        df['cde_frac'] = 0 if max_cde == 0 else df['ftsearch_cde']/max_cde
        df['dec_frac'] = 0 if max_dec == 0 else df['ftsearch_dec']/max_dec
        df['que_frac'] = 0 if max_que == 0 else df['ftsearch_question']/max_que
        df['syn_classsum_frac'] = 0 if max_syn_classsum == 0 else df['syn_classsum']/max_syn_classsum
        df['syn_propsum_frac'] = 0 if max_syn_propsum == 0 else df['syn_propsum']/max_syn_propsum
        df['syn_objsum_frac'] = 0 if max_syn_objsum == 0 else df['syn_objsum']/max_syn_objsum
        df['syn_classmax_frac'] = 0 if max_syn_classmax == 0 else df['syn_classmax']/max_syn_classmax
        df['syn_propmax_frac'] = 0 if max_syn_propmax == 0 else df['syn_propmax']/max_syn_propmax
        df['syn_objmax_frac'] = 0 if max_syn_objmax == 0 else df['syn_objmax']/max_syn_objmax
        df['enum_concept_frac'] = 0 if max_enum_concept == 0 else df['enum_concept_search']/max_enum_concept
        df['enum_ans_frac'] = 0 if max_enum_ans == 0 else df['enum_answer_search']/max_enum_ans
        df['ans_score_frac'] = 0 if max_ans_score == 0 else df['answer_count_score']/max_ans_score
        df['val_score_frac'] = 0 if max_val_score == 0 else df['value_score']/max_val_score
        df['cde_norm'] = NO_VAR_VALUE if std_cde == 0 else (df['ftsearch_cde'] - mean_cde)/std_cde
        df['dec_norm'] = NO_VAR_VALUE if std_dec == 0 else (df['ftsearch_dec'] - mean_dec)/std_dec
        df['syn_classsum_norm'] = NO_VAR_VALUE if std_syn_classsum == 0 else (df['syn_classsum'] - mean_syn_classsum)/std_syn_classsum
        df['syn_propsum_norm'] = NO_VAR_VALUE if std_syn_propsum == 0 else (df['syn_propsum'] - mean_syn_propsum)/std_syn_propsum
        df['syn_objsum_norm'] = NO_VAR_VALUE if std_syn_objsum == 0 else (df['syn_objsum'] - mean_syn_objsum)/std_syn_objsum
        df['syn_classmax_norm'] = NO_VAR_VALUE if std_syn_classmax == 0 else (df['syn_classmax'] - mean_syn_classmax)/std_syn_classmax
        df['syn_propmax_norm'] = NO_VAR_VALUE if std_syn_propmax == 0 else (df['syn_propmax'] - mean_syn_propmax)/std_syn_propmax
        df['syn_objmax_norm'] = NO_VAR_VALUE if std_syn_objmax == 0 else (df['syn_objmax'] - mean_syn_objmax)/std_syn_objmax
        df['question_norm'] = NO_VAR_VALUE if std_question == 0 else (df['ftsearch_question'] - mean_question)/std_question
        df['enum_concept_norm'] = NO_VAR_VALUE if std_enum_concept == 0 else (df['enum_concept_search'] - mean_enum_concept)/std_enum_concept
        df['enum_ans_norm'] = NO_VAR_VALUE if std_enum_ans == 0 else (df['enum_answer_search'] - mean_enum_ans)/std_enum_ans
        df['ans_score_norm'] = NO_VAR_VALUE if std_ans_score == 0 else (df['answer_count_score'] - mean_ans_score)/std_ans_score
        df['val_score_norm'] = NO_VAR_VALUE if std_val_score == 0 else (df['value_score'] - mean_val_score)/std_val_score
        df['logn'] = np.log(n)
        df['index'] = df['index'].astype('int')
        for col in df.columns:
            df.loc[df[col] != df[col],col] = 0
    return df

def append_target_metrics(
    column_expanded_X,
    annotated_result,
    g
):
    df = column_expanded_X.copy()
    if df.shape[0] > 0:
        annotated_cde = score_functions.get_de_id(annotated_result)
        if annotated_cde is not None:
            annotated_cde = int(annotated_cde)
        df['metric1'] = [score_functions.WEIGHTS['de_wt'] if i == annotated_cde else 0 for i in df['cde_id'].values]
        metric2_df = pd.DataFrame(
            score_functions.score_multiple_concept_overlap(annotated_cde,df['index'].values,g)
        )
        df = pd.merge(df,metric2_df,on='index',how='outer')
        df.loc[df['metric2'] != df['metric2'],'metric2'] = 0
        max_metric2 = max(df['metric2'])
        n = df.shape[0]
        df['metric2_max'] = max_metric2
        df['pct_metric2_pos'] = sum(df['metric2'] > 0)/n
        df['metric2_frac'] = 0 if max_metric2 == 0 else df['metric2']/max_metric2
        df['metric3'] = (df['metric1'] > 1).astype('int')
        df['metric4'] = (df['metric2'] > 0).astype('int')
        df['metric5'] = (df['metric2_frac'] > 0.5).astype('int')
    return df



def build_expanded_X(
    col_series,
    g, 
    NAMEINDEX_SEARCH_REQD = 25,
    NAMEINDEX_CDE_REQD = 5,
    MIN_SCORE = 0
):
    df = build_initial_column_data(
        col_series,
        g, 
        NAMEINDEX_SEARCH_REQD,
        NAMEINDEX_CDE_REQD,
        MIN_SCORE
    )
    df_expanded = expand_column_X(df)
    return df_expanded


def build_column_training_data(
    col_series,
    annotated_result,
    g, 
    NAMEINDEX_SEARCH_REQD = 25,
    NAMEINDEX_CDE_REQD = 5,
    MIN_SCORE = 0
):
    expanded_X = build_expanded_X(
        col_series,
        g,
        NAMEINDEX_SEARCH_REQD,
        NAMEINDEX_CDE_REQD,
        MIN_SCORE
    )
    training_X = append_target_metrics(
        expanded_X,
        annotated_result,
        g
    )
    return training_X
