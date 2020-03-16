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

def lower_upper_split(input_str):
    o = re.sub(r"([a-z])([A-Z])",r"\1 \2",input_str)
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

