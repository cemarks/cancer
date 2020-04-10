from neo4j import GraphDatabase
import re

def neo4j_connect(
    DATABASE_URI = "bolt://localhost:7687",
    DATABASE_USER = "neo4j",
    DATABASE_PASSWORD = "loHmjZWp"
):
    graphDB = GraphDatabase.driver(
        DATABASE_URI,
        auth = (
            DATABASE_USER,
            DATABASE_PASSWORD
        ),
        database="system",
        encrypted=False
    )
    return graphDB

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

def lower_upper_split(input_str):
    o = re.sub(r"([a-z])([A-Z])",r"\1 \2",input_str)
    o = re.sub(r"([0-9])([A-Z,a-z])",r"\1 \2",o)
    return o

def underscore_replace(input_str):
    o = input_str.replace("_"," ")
    return o

def period_replace(input_str):
    o = input_str.replace("."," ")
    return o

def query_graph(input_str,g):
    with g.session() as q:
        o = q.run(input_str)
    return o

def clean_string_for_fulltext(input_string):
    s = str(input_string).replace("/"," ").replace(","," ").replace("+"," ").replace("-"," ").replace("("," ").replace(")"," ").replace("["," ").replace("]"," ")
    return s.lower()


## Finding synonym parses of a string

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

def col_unique_values(col_series):
    no_nan = col_series.loc[col_series==col_series]
    unique_values = no_nan.unique().tolist()
    return unique_values

def is_not_nan(whatever):
    if whatever == whatever:
        return True
    else:
        return False
