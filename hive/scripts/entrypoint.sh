#!/bin/sh

HIVE_VERSION=4.0.1
HIVE_HOME=/opt/apache-hive-${HIVE_VERSION}-bin

# Check if schema exists
${HIVE_HOME}/bin/schematool -dbType mysql -info

if [ $? -eq 1 ]; then
    echo "Getting schema info failed. Probably not initialized. Initializing..."
    ${HIVE_HOME}/bin/schematool -initSchema -dbType mysql
fi

${HIVE_HOME}/bin/hive --service metastore