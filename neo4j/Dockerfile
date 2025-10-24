FROM neo4j:2025.05.0

# environment options
ENV NEO4J_PLUGINS='["apoc"]'
ENV NEO4J_dbms_security_procedures_unrestricted=apoc.*
ENV NEO4J_AUTH="neo4j/REPLACE_WITH_PASSWORD"

# move the dump into the image
COPY neo4j.dump /var/lib/neo4j/xfer/neo4j.dump
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["neo4j"]
