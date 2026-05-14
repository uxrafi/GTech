from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jCapabilityRegistry:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_all_agents(self):
        """Return all agents with their descriptions."""
        with self.driver.session() as session:
            result = session.run("MATCH (a:Agent) RETURN a.name AS name, a.description AS description")
            return [{"name": record["name"], "description": record["description"]} for record in result]

    def find_agents_by_capability(self, capability_keyword: str):
        """Find agents that provide a capability whose ID or description contains the keyword."""
        query = """
        MATCH (a:Agent)-[:PROVIDES]->(c:Capability)
        WHERE c.id CONTAINS $keyword OR c.description CONTAINS $keyword
        RETURN a.name AS agent_name, a.description AS agent_desc, c.id AS capability_id, c.description AS capability_desc
        """
        with self.driver.session() as session:
            result = session.run(query, keyword=capability_keyword)
            records = []
            for record in result:
                records.append({
                    "agent_name": record["agent_name"],
                    "agent_description": record["agent_desc"],
                    "capability_id": record["capability_id"],
                    "capability_description": record["capability_desc"]
                })
            return records

    def find_agents_for_task(self, task_description: str):
        """
        Use a simple keyword match on capability IDs and descriptions.
        For your thesis, you can later replace this with LLM‑based matching or graph traversal.
        """
        # Extract keywords from task (simple split)
        keywords = task_description.lower().split()
        # We'll match any capability whose id or description contains any of the keywords
        matched = []
        for kw in keywords:
            results = self.find_agents_by_capability(kw)
            for r in results:
                if r not in matched:
                    matched.append(r)
        return matched