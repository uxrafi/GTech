import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnector:
    def __init__(self):
        self.backend = os.getenv("DB_BACKEND", "neo4j").lower()
        self._connect()

    def _connect(self):
        if self.backend == "neo4j":
            from neo4j import GraphDatabase
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"Connected to Neo4j at {uri}")
        elif self.backend == "postgres":
            import psycopg2
            dsn = os.getenv("POSTGRES_DSN")
            self.conn = psycopg2.connect(dsn)
            print(f"Connected to PostgreSQL")
        else:
            raise ValueError(f"Unsupported DB_BACKEND: {self.backend}")

    def close(self):
        if self.backend == "neo4j":
            self.driver.close()
        elif self.backend == "postgres":
            self.conn.close()

    # Simple Cypher / SQL wrappers – extend as needed
    def run(self, query, parameters=None):
        parameters = parameters or {}
        if self.backend == "neo4j":
            with self.driver.session() as session:
                return session.run(query, parameters).data()
        elif self.backend == "postgres":
            with self.conn.cursor() as cur:
                cur.execute(query, parameters)
                if cur.description:
                    return cur.fetchall()
                else:
                    self.conn.commit()
                    return []

if __name__ == "__main__":
    db = DatabaseConnector()
    # Example: print version
    if db.backend == "neo4j":
        print(db.run("RETURN 'Database ready' as msg"))
    else:
        print(db.run("SELECT version()"))
    db.close()