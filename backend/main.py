from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback
import uvicorn

from routers import training, datasets, experiments, progress, replay

app = FastAPI(
    title="Neural Forge API",
    description=(
        "Backend for NEURAL FORGE — a gamified, educational neural network platform. "
        "All ML logic runs on pure NumPy. No external ML frameworks required."
    ),
    version="1.0.0",
)

# ── CORS — allow all origins for Streamlit frontend ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(training.router,    prefix=API_PREFIX)
app.include_router(datasets.router,    prefix=API_PREFIX)
app.include_router(experiments.router, prefix=API_PREFIX)
app.include_router(progress.router,    prefix=API_PREFIX)
app.include_router(replay.router,      prefix=API_PREFIX)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "service": "neural-forge-backend",
        "version": "1.0.0",
    }


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Neural Forge API is running.",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": API_PREFIX,
    }


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "traceback": tb,
        },
    )
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)