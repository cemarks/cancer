#!/usr/bin/python3

import os
import json
from my_modules import score_functions,utils


DATABASE_URI = "bolt://localhost:7688"
DATABASE_USER = "neo4j"
DATABASE_PASSWORD = "loHmjZWp"
# ANNOTATION_FILE = "/home/cemarks/Projects/cancer/data/Data/training-col100_annotated/Annotated-table-125633.947046.json"
# SUBMISSION_FILE = "/home/cemarks/Projects/cancer/mount_folder/output/table-125633.947046-Submission.json"
# ANNOTATION_FILE = "/home/cemarks/Projects/cancer/data/Data/training-col100_annotated/Annotated-table-130518.468467.json"
# SUBMISSION_FILE = "/home/cemarks/Projects/cancer/mount_folder/output/table-130518.468467-Submission.json"

SUBMISSION_DIR = "/home/cemarks/Projects/cancer/mount_folder/output"
ANNOTATION_DIR = "/home/cemarks/Projects/cancer/data/leaderboard"

submission_files = os.listdir(SUBMISSION_DIR)

graphDB = utils.neo4j_connect(
    DATABASE_URI,
    DATABASE_USER,
    DATABASE_PASSWORD
)



for sf in submission_files:
    file_base = sf.rstrip("-Submission.json")
    ANNOTATION_FILE = os.path.join(
        ANNOTATION_DIR,
        "Annotated-" + file_base + ".json"
    )
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR,sf)
    with open(ANNOTATION_FILE,'r') as f:
        annotation_json = json.load(f)
    with open(SUBMISSION_FILE,'r') as f:
        submission_json = json.load(f)
    score = score_functions.score_submission(
        submission_json,
        annotation_json,
        graphDB
    )
    print(sf)
    print(score)
    print()


