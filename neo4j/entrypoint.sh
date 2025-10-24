#!/bin/bash
set -e

# for some reason running it during the container build doesn't work so we
# need to wrap the entrypoint to run this before it starts
neo4j-admin database load neo4j \
  --from-path=/var/lib/neo4j/xfer \
  --overwrite-destination=true

# now delegate to the original Neo4j entrypoint
exec /startup/docker-entrypoint.sh "$@"