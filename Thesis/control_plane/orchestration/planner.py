"""
Meta‑Cognitive Orchestrator for Self‑Adaptive Agentic Systems
Thesis: Umar Rafi, Georgia Tech OMSCS

This module implements a planner that:
- Takes a high‑level natural language goal.
- Uses an LLM (or mock) to decompose the goal into a sequence of tasks.
- Queries a capability registry (mock or database) to find agents that can execute each task.
- Builds an executable workflow plan.
"""

import json
import re
from typing import Dict, List, Any, Optional
from ai.llm.client import LLMClient


class CapabilityRegistry:
    """
    Registry of available agents/capabilities.
    Replace with Neo4j or PostgreSQL implementation later.
    """

    def __init__(self):
        # Mock capabilities for financial log intelligence
        self.capabilities = [
            {
                "name": "TransactionLogParser",
                "description": "Parses raw transaction logs into structured format",
                "inputs": ["raw_transaction_logs"],
                "outputs": ["parsed_transactions"],
                "domains": ["transaction", "parsing"]
            },
            {
                "name": "AnomalyDetector",
                "description": "Detects anomalies in parsed transaction data using statistical methods",
                "inputs": ["parsed_transactions"],
                "outputs": ["anomaly_scores", "flagged_transactions"],
                "domains": ["anomaly_detection", "fraud"]
            },
            {
                "name": "AlertCorrelator",
                "description": "Correlates security alerts with system events",
                "inputs": ["security_alerts", "system_events"],
                "outputs": ["correlated_alerts"],
                "domains": ["correlation", "security"]
            },
            {
                "name": "ComplianceValidator",
                "description": "Validates log data against regulatory rules (e.g., SOX, GDPR)",
                "inputs": ["parsed_transactions", "compliance_rules"],
                "outputs": ["compliance_reports", "violations"],
                "domains": ["compliance", "reporting"]
            },
            {
                "name": "FraudPatternMiner",
                "description": "Mines frequent patterns from transaction data to detect fraud",
                "inputs": ["parsed_transactions"],
                "outputs": ["fraud_patterns", "risk_scores"],
                "domains": ["pattern_mining", "fraud"]
            }
        ]

    def find_capabilities(self, required_capability: str, domain_hint: str = None) -> List[Dict]:
        """
        Search for capabilities by keyword (simple match).
        For thesis: replace with vector similarity or graph traversal.
        """
        results = []
        required_lower = required_capability.lower()
        for cap in self.capabilities:
            name_match = required_lower in cap["name"].lower()
            desc_match = required_lower in cap["description"].lower()
            domain_match = domain_hint and any(domain in cap["domains"] for domain in [domain_hint])
            if name_match or desc_match or domain_match:
                results.append(cap)
        return results

    def get_all(self) -> List[Dict]:
        return self.capabilities


class WorkflowPlanner:
    """
    Orchestrator that plans and composes workflows.
    """

    def __init__(self, llm_client: LLMClient, registry: CapabilityRegistry):
        self.llm = llm_client
        self.registry = registry

    def decompose_goal(self, goal: str) -> List[str]:
        """
        Use LLM to break a high‑level goal into a sequence of subtasks.
        Returns a list of task descriptions (e.g., ["parse logs", "detect anomalies"]).
        """
        decomposition_prompt = f"""
You are a workflow planner for financial log intelligence. Given a high-level goal, break it down into a sequence of concrete subtasks. Each subtask should be a short action phrase (e.g., "parse transaction logs", "detect anomalies", "correlate alerts").

Output only a JSON list of strings. No extra text.

Goal: {goal}
"""
        response = self.llm.generate(decomposition_prompt, system="You are an expert workflow planner.")
        try:
            # Extract JSON from response (handle markdown or plain)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group())
            else:
                # Fallback: split by lines
                tasks = [line.strip("- ").strip() for line in response.split("\n") if line.strip() and not line.startswith("```")]
            if not tasks:
                raise ValueError("No valid tasks found")
            return tasks
        except Exception as e:
            print(f"Failed to parse LLM decomposition: {e}\nResponse: {response}")
            # Return a default decomposition for the demo
            return ["parse transaction logs", "detect anomalies"]

    def map_tasks_to_capabilities(self, tasks: List[str]) -> List[Dict]:
        """
        For each task, find matching agents from the capability registry.
        Returns a list of plan steps, each with 'task_description' and 'selected_capability'.
        """
        plan = []
        for task in tasks:
            # Simple keyword extraction: take first few words
            keywords = task.lower().split()[:4]
            matched = None
            for cap in self.registry.get_all():
                cap_text = (cap["name"] + " " + cap["description"]).lower()
                if any(keyword in cap_text for keyword in keywords):
                    matched = cap
                    break
            if not matched:
                # Try exact match on name
                for cap in self.registry.get_all():
                    if cap["name"].lower() in task.lower():
                        matched = cap
                        break
            if not matched:
                # Fallback: use first capability that covers relevant domain
                if "anomaly" in task or "fraud" in task:
                    matched = next((c for c in self.registry.get_all() if "anomaly" in c["domains"]), None)
                elif "parse" in task or "log" in task:
                    matched = next((c for c in self.registry.get_all() if "pars" in c["name"].lower()), None)
                elif "compliance" in task:
                    matched = next((c for c in self.registry.get_all() if "compliance" in c["name"].lower()), None)

            plan.append({
                "task_description": task,
                "selected_capability": matched if matched else {"name": "Unknown", "description": "No matching capability found"}
            })
        return plan

    def generate_workflow(self, goal: str) -> Dict[str, Any]:
        """
        Full orchestration: decompose goal, map to capabilities, produce workflow.
        """
        print(f"[Orchestrator] Decomposing goal: {goal}")
        tasks = self.decompose_goal(goal)
        print(f"[Orchestrator] Decomposed into {len(tasks)} tasks: {tasks}")
        plan = self.map_tasks_to_capabilities(tasks)
        workflow = {
            "goal": goal,
            "tasks": tasks,
            "plan": plan,
            "status": "ok"
        }
        return workflow

    def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate execution of the workflow (for demonstration).
        In your thesis, this would call actual agent APIs or run the plan.
        """
        results = []
        for step in workflow["plan"]:
            cap = step["selected_capability"]
            if cap["name"] == "Unknown":
                results.append({"step": step["task_description"], "status": "failed", "reason": "No capability found"})
            else:
                results.append({"step": step["task_description"], "status": "success", "agent": cap["name"]})
        workflow["execution_results"] = results
        return workflow


# ========== DEMO ==========
if __name__ == "__main__":
    # Initialize LLM client (mock or real)
    llm = LLMClient()
    registry = CapabilityRegistry()
    orchestrator = WorkflowPlanner(llm, registry)

    # Example high‑level goal
    test_goal = "Detect anomalies in payment transactions and flag potential fraud"

    # Generate workflow
    workflow = orchestrator.generate_workflow(test_goal)
    print("\n--- Generated Workflow ---")
    print(json.dumps(workflow, indent=2))

    # Simulate execution
    execution = orchestrator.execute_workflow(workflow)
    print("\n--- Execution Results ---")
    print(json.dumps(execution, indent=2))