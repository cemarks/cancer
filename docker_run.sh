docker run --rm --name cancer \
-v ${HOME}/Projects/cancer/mount_folder/input:/input \
-v ${HOME}/Projects/cancer/mount_folder/output:/output \
-v ${HOME}/Projects/cancer/mount_folder/data:/data \
docker.synapse.org/syn21566013/madc-python-neo4j:${1}
