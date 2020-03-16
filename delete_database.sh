PROJECT_DIR=${HOME}/Projects/cancer
NEO4J_BIN=${PROJECT_DIR}/docker-python/neo4j-community-4.0.0/bin
NEO4J_DATA=${PROJECT_DIR}/docker-python/neo4j-community-4.0.0/data


${NEO4J_BIN}/neo4j stop
rm -rf ${NEO4J_DATA}/databases
rm -rf ${NEO4J_DATA}/transactions
${NEO4J_BIN}/neo4j start
