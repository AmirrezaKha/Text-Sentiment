FROM bitnami/spark:3.4.1

USER root

RUN apt update && apt install -y wget

RUN wget https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.4_2.12/1.3.1/iceberg-spark-runtime-3.4_2.12-1.3.1.jar -P /opt/bitnami/spark/jars

RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.5/hadoop-aws-3.3.5.jar -P /opt/bitnami/spark/jars

RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.901/aws-java-sdk-bundle-1.11.901.jar -P /opt/bitnami/spark/jars

RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.5/hadoop-common-3.3.5.jar -P /opt/bitnami/spark/jars

ENV MLFLOW_TRACKING_URI=http://localhost:5000

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN python -m spacy download en_core_web_sm

RUN pip install mlflow

RUN rm /tmp/requirements.txt

USER 1001
