FROM python:3.7
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean;
RUN pip install neo4j
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install stringdist

RUN mkdir /input
RUN mkdir /output
RUN mkdir /data

# This neo4j folder contains a pre-built graph database
COPY neo4j-community-4.0.0 /neo4j
ENV PATH=${PATH}:/neo4j/bin
ENV PYTHONPATH=/meta_challenge_py

COPY meta_challenge_py /meta_challenge_py
COPY exec.sh /
COPY models /models

# Make neo4j accessible from outside container
# RUN sed -i "s/# dbms.connectors.default_listen_address=0.0.0.0/dbms.connectors.default_listen_address=0.0.0.0/" /neo4j/conf/neo4j.conf


ENTRYPOINT ["sh","/exec.sh"]

# CMD sh

