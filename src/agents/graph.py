"""
src/agents/graph.py
-------------------
Neo4jGraph singleton.  One connection, shared by all tools.

Also exposes get_driver() for tools that need raw neo4j.Driver
(e.g. direct Cypher queries outside LangChain chains).

.env keys:
  NEO4J_URI       neo4j+s://<id>.databases.neo4j.io
  NEO4J_USER      neo4j
  NEO4J_PASS      <password>
  NEO4J_DATABASE  neo4j   (Aura Free always uses 'neo4j')
"""

from __future__ import annotations

import logging
import os

import dotenv
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

_URI      = os.getenv("NEO4J_URI", "")
_USER     = os.getenv("NEO4J_USER", "neo4j")
_PASS     = os.getenv("NEO4J_PASS", "")
_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def _build_graph() -> Neo4jGraph:
    logger.info("Connecting Neo4jGraph to %s", _URI[:40])
    return Neo4jGraph(
        url=_URI,
        username=_USER,
        password=_PASS,
        database=_DATABASE,
        enhanced_schema=False,   # skip expensive schema refresh on every import
    )


def get_driver():
    """Raw neo4j.Driver for tools that run plain Cypher outside LangChain."""
    return GraphDatabase.driver(_URI, auth=(_USER, _PASS))


# Singletons
graph  = _build_graph()
driver = get_driver()