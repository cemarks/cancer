cp /home/cemarks/Projects/cancer/sandbox/nomatch_model.pkl /home/cemarks/Projects/cancer/docker-python/models/
cp /home/cemarks/Projects/cancer/sandbox/value_regression.pkl /home/cemarks/Projects/cancer/docker-python/models/
docker build -t docker.synapse.org/syn21566013/madc-python-neo4j:${1} ./docker-python
docker run --rm -v /home/cemarks/Projects/cancer/mount_folder/input:/input -v /home/cemarks/Projects/cancer/mount_folder/output:/output -v /home/cemarks/Projects/cancer/mount_folder/data:/data docker.synapse.org/syn21566013/madc-python-neo4j:${1} $2
