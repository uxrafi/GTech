from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from ai.llm.client import LLMClient
from control_plane.orchestration.planner import WorkflowPlanner, CapabilityRegistry

router = APIRouter()


class ExecutionRequest(BaseModel):
    goal: str


@router.post("/run")
def run_workflow(request: ExecutionRequest) -> Dict[str, Any]:
    llm = LLMClient()
    registry = CapabilityRegistry()
    planner = WorkflowPlanner(llm_client=llm, registry=registry)

    workflow = planner.generate_workflow(request.goal)
    execution = planner.execute_workflow(workflow)

    return execution