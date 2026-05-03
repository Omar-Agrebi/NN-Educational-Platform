from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers import training, datasets, experiments, progress, replay, custom_dataset

app = FastAPI(title="Neural Forge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(experiments.router, prefix="/api/v1")
app.include_router(progress.router, prefix="/api/v1")
app.include_router(replay.router, prefix="/api/v1")
app.include_router(custom_dataset.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "neural-forge-backend"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
