from fastapi import FastAPI

from apps.api.routes.workflows import router as workflow_router
from apps.api.routes.executions import router as execution_router

app = FastAPI(
    title="Regulated AI Control Plane",
    version="0.1.0"
)

app.include_router(
    workflow_router,
    prefix="/workflow",
    tags=["workflow"]
)

app.include_router(
    execution_router,
    prefix="/execution",
    tags=["execution"]
)


@app.get("/")
def root():
    return {
        "message": "Regulated AI Control Plane API running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }