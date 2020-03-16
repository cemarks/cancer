#!/bin/bash

source ${HOME}/Projects/cancer/delete_database.sh
sleep 10
source ${HOME}/Projects/cancer/env/bin/activate
python ${HOME}/Projects/cancer/docker-python/meta_challenge_py/my_modules/neo_create.py ${HOME}/Projects/cancer/data/Data/reference
deactivate
