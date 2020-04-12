#!/usr/bin/python3

from my_modules import utils, datachecks
import stringdist



def classify_values(col_values,cde_index,g):
    query = "MATCH (n:CDE) where ID(n) = {0:d} RETURN n.DATATYPE, n.VALUE_DOMAIN_TYPE, n.DISPLAY_FORMAT, n.CDE_ID".format(int(cde_index))
    result = utils.query_graph(query,g)
    values = result.values()
    value_domain_attributes = values[0]
    output_list = []
    if value_domain_attributes[1]=='Enumerated':
        for v in col_values:
            classification_dict = classify_single_enum_value(v,int(cde_index),g)
            output_list.append(classification_dict)
    elif value_domain_attributes[2] is not None:
        for v in col_values:
            classification_dict = classify_display_value(v,value_domain_attributes[2])
            output_list.append(classification_dict)
    else:
        for v in col_values:
            classification_dict = classify_datatype(v,value_domain_attributes[2])
            output_list.append(classification_dict)
    return (output_list)

def classify_single_enum_value(col_value,cde_index,g, SEARCH_CUTOFF = 0):
    output_dict = {'observedValue':str(col_value),'permissibleValue':{}}
    query = "CALL db.index.fulltext.queryNodes(\"ansindex\",\"{0:s}\") YIELD node as a, score ".format(str(col_value))
    query += "MATCH (n:CDE) - [:PERMITS] - (ans:Answer) - [:CAN_BE] - (a:AnswerText) WHERE ID(n) = {0:d} ".format(cde_index)
    query += "RETURN ID(ans), 'Answer', score, a.name"
    result = utils.query_graph(query,g)
    answer_values = result.values()
    query = "CALL db.index.fulltext.queryNodes(\"nameindex\",\"{0:s}\") YIELD node as s, score ".format(str(col_value))
    query += "MATCH (n:CDE) - [:PERMITS] - (ans:Answer) - [:EQUALS] - (con:Concept) - [:IS_CALLED] - (s:Synonym) WHERE ID(n) = {0:d} ".format(cde_index)
    query += "RETURN ID(ans), 'Synonym', score, s.name, con.CODE"
    result = utils.query_graph(query,g)
    syn_values = result.values()
    all_results = answer_values + syn_values
    all_results = [i for i in all_results if i[2] > SEARCH_CUTOFF]
    if len(all_results) > 0:
        all_results.sort(key=lambda z: z[2],reverse=True)
        ans_index = all_results[0][0]
        ans_results = [i for i in all_results if i[0] == ans_index]
        # Now we need to choose the best synonym
        synonyms = [i for i in ans_results if i[1] == 'Synonym']
        if len(synonyms) > 0:  #Choose the best based on 1: search score, and 2: stringdist
            synonyms.sort(key=lambda z: (-z[2],stringdist.levenshtein_norm(str(z[3]).lower(),str(col_value).lower())))
            output_dict['permissibleValue']['value'] = str(synonyms[0][3])
            output_dict['permissibleValue']['conceptCode'] = 'ncit:' + str(synonyms[0][4])
        else:
            query = "MATCH (a:Answer) - [:EQUALS] - (c:Concept) - [:IS_CALLED] - (s:Synonym) WHERE ID(a) = {0:d} RETURN c.CODE,s.name".format(ans_index)
            result = utils.query_graph(query,g)
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
    check = datachecks.check_display_format(str(col_value),str(display_format))
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
    check = datachecks.check_datatype(str(col_value),str(datatype))
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


