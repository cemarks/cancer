#!/usr/bin/python3

import json
from my_modules import utils
import numpy as np

# Score functions


WEIGHTS = {
    'de_wt': 1.5,
    'dec_wt': 1.0,
    'vd_wt': 0.5,
    'top_wt': 2.0
}


def read_annotation(full_file_path):
    with open(full_file_path,'r') as f:
        annotated = json.load(f)
    return annotated

def get_col_result_annotation(annotated,column_no):
    col_index = [ann_col['columnNumber'] for ann_col in annotated['columns']].index(column_no)
    return annotated['columns'][col_index]['results'][0]['result']

def get_de_id(res_annotation):
    return res_annotation['dataElement']['id']

def get_de_concepts(cde_id,g):
    query = "MATCH (n:CDE) - [:IS_CAT] - (d:DEC) - [] - (c:Concept) WHERE n.CDE_ID = {0:d} RETURN c.CODE".format(cde_id)
    codes = utils.query_graph(query,g).value()
    return codes

def get_de_concepts_list(cde_index_list,g):
    query = "MATCH (n:CDE) - [:IS_CAT] - (d:DEC) - [] - (c:Concept) WHERE ID(n) IN [{0:s}] ".format(",".join([str(i) for i in cde_index_list]))
    query += "RETURN ID(n),COLLECT(c.CODE)"
    codes = utils.query_graph(query,g).values()
    return codes

def score_concept_overlap(cde_id1,cde_id2,g):
    codes_1 = get_de_concepts(cde_id1,g)
    codes_2 = get_de_concepts(cde_id2,g)
    j = jaccard_dist(codes_1,codes_2)
    return j

def score_multiple_concept_overlap(annotated_cde_id,cde_index_list,g):
    codes_2 = get_de_concepts_list(cde_index_list,g)
    if annotated_cde_id is not None:
        codes_1 = get_de_concepts(annotated_cde_id,g)
        out_dict = {
            'index': [i[0] for i in codes_2],
            'metric2': [jaccard_dist(codes_1,i[1]) for i in codes_2]
        }
    else:
        out_dict = {
            'index': [i[0] for i in codes_2],
            'metric2': [0 for i in codes_2]
        }
    return out_dict

def score_cde(cde_id1,cde_id2):
    if cde_id1==cde_id2:
        score = 1
    else:
        score = 0
    return score

def score_value_domain(submitted_vd,annotated_res,g):
    annotated_vd = annotated_res['valueDomain']
    annotated_cde_id = annotated_res['dataElement']['id']
    query = 'MATCH (n:CDE) WHERE n.CDE_ID = {0:d} RETURN n.VALUE_DOMAIN_TYPE'.format(annotated_cde_id)
    result = utils.query_graph(query,g)
    value = result.value()
    if len(value) == 0:
        score = 0
    else:
        submitted_observed_values = [i['observedValue'] for i in submitted_vd]
        enumerated = value[0] == "Enumerated"
        if enumerated:
            check_column = 'value'
        else:
            check_column = 'conceptCode'
        if len(annotated_vd) > 0:
            mismatch_count = 0
            for annotated_value_dict in annotated_vd:
                if annotated_value_dict['observedValue'] in submitted_observed_values:
                    submitted_value_dict = submitted_vd[submitted_observed_values.index(annotated_value_dict['observedValue'])]
                    if submitted_value_dict['permissibleValue'][check_column] != annotated_value_dict['permissibleValue'][check_column]:
                        mismatch_count += 1
                else:
                    mismatch_count += 1
            score = 1-mismatch_count / len(annotated_vd)
        else:
            score = 0
    return score

def score_result(submitted_result_dict,annotated_result_dict,g):
    if submitted_result_dict['dataElement']['name'] == 'NOMATCH':
        if annotated_result_dict['dataElement']['name'] == 'NOMATCH':
            metric1 = 1
            metric2 = 1
            metric3 = 1
        else:
            metric1 = 0
            metric2 = 0
            metric3 = 0
    elif annotated_result_dict['dataElement']['name'] == 'NOMATCH':
        metric1 = 0
        metric2 = 0
        metric3 = 0
    else:
        sub_cde_id = get_de_id(submitted_result_dict)
        ann_cde_id = get_de_id(annotated_result_dict)
        metric1 = score_cde(sub_cde_id,ann_cde_id)
        if metric1 == 1:
            metric2 = 1
        else:
            metric2 = score_concept_overlap(sub_cde_id,ann_cde_id,g)
        metric3 = score_value_domain(submitted_result_dict['valueDomain'],annotated_result_dict,g)
    return metric1,metric2,metric3


def score_column(submitted_col_dict,annotation_json,g):
    column_no = submitted_col_dict['columnNumber']
    annotated_result_dict = get_col_result_annotation(annotation_json,column_no)
    col_results = submitted_col_dict['results']
    scores = []
    for r in col_results:
        de_score,dec_score,vd_score = score_result(r['result'],annotated_result_dict,g)
        initial_score = WEIGHTS['de_wt'] * de_score + WEIGHTS['dec_wt'] * dec_score + WEIGHTS['vd_wt'] * vd_score
        additional = (WEIGHTS['top_wt'] + 1 - r['resultNumber']) * np.mean([de_score,dec_score,vd_score])
        scores.append(initial_score + additional)
    return(np.max(scores))

def score_submission(submitted_json,annotation_json,g):
    scores = []
    for submitted_col in submitted_json['columns']:
        s = score_column(submitted_col,annotation_json,g)
        scores.append(s)
    return np.mean(scores)








def jaccard_dist(list_1,list_2):
    set1 = set(list_1)
    set2 = set(list_2)
    i = set1.intersection(set2)
    u = set1.union(set2)
    l_intersect = len(i)
    l_union = len(u)
    if (l_intersect == 0) and (l_union == 0):
        out = 1
    elif (l_intersect ==0) or (l_union == 0):
        out = 0
    else:
        out = l_intersect / l_union
    return out



# def metric1(node_ind,...)