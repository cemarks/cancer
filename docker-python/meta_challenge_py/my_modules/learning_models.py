#!/usr/bin/python3


from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def lr_transform(x):
    predictor_columns = [
        # "max_cde",
        # "max_dec",
        # "max_que",
        # "max_syn_classsum",
        # "max_syn_propsum",
        # "max_syn_objsum",
        # "max_syn_classmax",
        "max_syn_propmax",
        # "max_syn_objmax",
        "max_enum_concept",
        # "max_enum_ans",
        # "max_ans_score",
        # "max_val_score",
        # "max_secondary_search",
        # "pct_cde",
        # "pct_dec",
        # "pct_que",
        # "pct_syn_classsum",
        "pct_syn_propsum",
        # "pct_syn_objsum",
        # "pct_syn_classmax",
        # "pct_syn_propmax",
        # "pct_syn_objmax",
        # "pct_enum_concept",
        # "pct_enum_ans",
        # "pct_ans_score",
        # "pct_val_score",
        "pct_secondary_search",
        # "n",
        "logn"
    ]
    x_copy = x.copy()
    x_copy.insert(
        x_copy.shape[1],
        'c1',
        x['pct_secondary_search'].multiply(x['max_enum_ans'])
    )
    x_copy.insert(
        x_copy.shape[1],
        'c2',
        x['pct_secondary_search'].multiply(x['max_val_score'])
    )
    x_copy.insert(
        x_copy.shape[1],
        'c3',
        x['pct_secondary_search'].multiply(x['pct_enum_concept'])
    )
    x_copy.insert(
        x_copy.shape[1],
        'c4',
        x['pct_secondary_search'].multiply(x['pct_que'])
    )
    x_copy.insert(
        x_copy.shape[1],
        'c5',
        x['pct_secondary_search'].multiply(x['pct_syn_propsum'] + x['pct_syn_objsum'])
    )
    x_copy.insert(
        x_copy.shape[1],
        'c6',
        x['pct_cde'] + x['pct_que']
    )
    x_copy.insert(
        x_copy.shape[1],
        'c7',
        np.sqrt((x['max_cde']+1).multiply(x['max_que']+1))-1
    )
    predictor_columns += ['c6','c7']
    z = x_copy[predictor_columns].values
    poly = PolynomialFeatures(degree = 2)
    z = poly.fit_transform(z)
    return(z)


def rr_transform(x):
    predictor_columns = [
        "secondary_search",
        "ftsearch_cde",
        # "ftsearch_dec",
        # "syn_classsum",
        # "syn_propsum",
        # "syn_objsum",
        # "syn_classmax",
        # "syn_propmax",
        # "syn_objmax",
        # "ftsearch_question",
        "enum_concept_search",
        # "enum_answer_search",
        # "answer_count_score",
        # "value_score",
        # "max_cde",
        # "max_dec",
        # "max_que",
        # "max_syn_classsum",
        # "max_syn_propsum",
        # "max_syn_objsum",
        # "max_syn_classmax",
        # "max_syn_propmax",
        # "max_syn_objmax",
        # "max_enum_concept",
        # "max_enum_ans",
        # "max_ans_score",
        # "max_val_score",
        # "max_secondary_search",
        # "pct_cde",
        # "pct_dec",
        "pct_que",
        # "pct_syn_classsum",
        # "pct_syn_propsum",
        # "pct_syn_objsum",
        # "pct_syn_classmax",
        # "pct_syn_propmax",
        # "pct_syn_objmax",
        # "pct_enum_concept",
        # "pct_enum_ans",
        # "pct_ans_score",
        # "pct_val_score",
        # "pct_secondary_search",
        # "cde_frac",
        # "dec_frac",
        # "que_frac",
        # "syn_classsum_frac",
        # "syn_propsum_frac",
        # "syn_objsum_frac",
        # "syn_classmax_frac",
        # "syn_propmax_frac",
        # "syn_objmax_frac",
        # "enum_concept_frac",
        # "enum_ans_frac",
        # "ans_score_frac",
        # "val_score_frac",
        "cde_norm",
        # "dec_norm",
        # "syn_classsum_norm",
        # "syn_propsum_norm",
        # "syn_objsum_norm",
        # "syn_classmax_norm",
        # "syn_propmax_norm",
        "syn_objmax_norm",
        "question_norm",
        # "enum_concept_norm",
        "enum_ans_norm",
        # "ans_score_norm",
        "val_score_norm",
        # "n",
        "logn"
    ]
    poly = PolynomialFeatures(degree = 2)
    Z_poly = poly.fit_transform(x[predictor_columns])
    # Z_poly = x[predictor_columns]
    return Z_poly
