from fastapi import APIRouter
from pydantic import BaseModel

from ai.llm.client import LLMClient
from control_plane.orchestration.planner import (
    WorkflowPlanner,
    CapabilityRegistry
)

router = APIRouter()


class WorkflowRequest(BaseModel):
    goal: str


@router.post("/generate")
def generate_workflow(request: WorkflowRequest):

    llm = LLMClient()

    registry = CapabilityRegistry()

    planner = WorkflowPlanner(
        llm_client=llm,
        registry=registry
    )

    workflow = planner.generate_workflow(request.goal)

    return workflow