#!/usr/bin/python3

import json
from my_modules import score_functions,utils


DATABASE_URI = "bolt://localhost:7688"
DATABASE_USER = "neo4j"
DATABASE_PASSWORD = "loHmjZWp"
ANNOTATION_FILE = "/home/cemarks/Projects/cancer/data/Data/training-col100_annotated/Annotated-table-130518.468467.json"
SUBMISSION_FILE = "/home/cemarks/Projects/cancer/mount_folder/output/table-130518.468467-Submission.json"

with open(ANNOTATION_FILE,'r') as f:
    annotation_json = json.load(f)

# with open(SUBMISSION_FILE,'r') as f:
#     submission_json = json.load(f)


with open(ANNOTATION_FILE,'r') as f:
    submission_json = json.load(f)

graphDB = utils.neo4j_connect(
    DATABASE_URI,
    DATABASE_USER,
    DATABASE_PASSWORD
)

score = score_functions.score_submission(
    submission_json,
    annotation_json,
    graphDB
)


print(score)