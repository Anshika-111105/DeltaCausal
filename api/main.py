from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import causal, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deltacausal_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DeltaCausal Uplift Engine API successfully initialized and ready for inference.")
    yield


app = FastAPI(
    title="DeltaCausal: Counterfactual Business Decision & Uplift Engine API",
    description=(
        "Production REST API for incremental causal effect estimation, "
        "uplift modeling, customer segmentation, and budget-constrained policy optimization."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(causal.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
