#!/usr/bin/python

from my_modules import value_classifiers, utils


def create_result_array(column_class_list,unique_values,g):
    results = []
    for i,cde_id in enumerate(column_class_list):
        results.append(
            create_single_result(
                result_no = i+1,
                cde_id = cde_id,
                unique_values = unique_values,
                graphdb = g
            )
        )
    return results


def create_nomatch_result(result_no,unique_values):
    vd = create_nomatch_value_domain(unique_values)
    r = {
        'resultNumber': result_no,
        'result':{
            'dataElement':{
                'id': None,
                'name': 'NOMATCH'
            },
            'dataElementConcept':{
                'id': None,
                'name': "NOMATCH",
                'conceptCodes': []
            },
            'valueDomain': vd
        }
    }
    return r

def create_nomatch_value_domain(unique_values):
    vd = [
        {
            'observedValue': str(i),
            'permissibleValue': {
                'value': 'NOMATCH',
                'conceptCode': None
            }
        } for i in unique_values
    ]
    return vd

def create_matched_value_domain(cde_id,unique_values,graphdb):
    if len(unique_values) == 0:
        return []
    else:
        q = "MATCH (n:CDE) WHERE n.CDE_ID = {0:d} RETURN ID(n)".format(cde_id)
        query_result = utils.query_graph(q,graphdb)
        cde_node_indices = query_result.value()
        candidates = []
        for cde_index in cde_node_indices:
            c = value_classifiers.classify_values(unique_values,cde_index,graphdb)
            neg_score = len([i for i in c if i['permissibleValue']['value'] in ['NOMATCH','NONCONFORMING']])
            candidates.append({
                'score':neg_score,
                'vd': c
            })
        candidates.sort(key=lambda z: z['score'])
        return candidates[0]['vd']

def create_matched_result(result_no,cde_id,unique_values,graphdb):
    vd = create_matched_value_domain(cde_id,unique_values,graphdb)
    q = "MATCH (n:CDE) - [:IS_CAT] - (d:DEC) WHERE n.CDE_ID = {0:d} RETURN n.CDE_LONG_NAME, d.DEC_ID, d.name".format(cde_id)
    query_result = utils.query_graph(q,graphdb)
    cde_data = query_result.values()[0]
    conceptcode_query = "MATCH (d:DEC) - [:IS_PROP] - (c:Concept) WHERE d.DEC_ID = {0:d} RETURN c.CODE as ConceptCode \n".format(cde_data[1])
    conceptcode_query += "UNION ALL MATCH (d:DEC) - [:IS_OBJ] - (c:Concept) WHERE d.DEC_ID = {0:d} RETURN c.CODE as ConceptCode ".format(cde_data[1])
    query_result = utils.query_graph(conceptcode_query,graphdb)
    conceptcodes = query_result.value()
    unique_concepts = list(set(conceptcodes))
    r = {
        'resultNumber': result_no,
        'result':{
            'dataElement':{
                'id': cde_id,
                'name': cde_data[0]
            },
            'dataElementConcept':{
                'id': cde_data[1],
                'name': cde_data[2],
                'conceptCodes': ["ncit:{0:s}".format(str(u)) for u in unique_concepts]
            },
            'valueDomain': vd
        }
    }
    return r

def create_single_result(result_no,cde_id,unique_values,graphdb):
    if str(cde_id).lower() == 'nomatch':
        r = create_nomatch_result(result_no,unique_values)
    else:
        r = create_matched_result(result_no,cde_id,unique_values,graphdb)
    return r





