# consyn/api/main.py
"""
FastAPI application for serving Consyn AI models.
This module provides a REST API for text generation using Consyn models.
"""

import os
import logging
import asyncio
from typing import List, Optional

from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .utils import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Consyn AI API",
    description="API for text generation with Consyn AI models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Store models to preload in app state
app.state.models_to_preload = []


@app.on_event("startup")
async def startup_preload_models():
    """Preload models on startup."""
    # Get models to preload
    models_to_preload = app.state.models_to_preload
    
    if not models_to_preload:
        # Check environment variable
        models_env = os.environ.get("MODELS_TO_PRELOAD")
        if models_env:
            models_to_preload = models_env.split(",")
        else:
            # Default to verse model
            models_to_preload = ["verse"]
            
    # Preload models
    for model_name in models_to_preload:
        try:
            logger.info(f"Preloading model: {model_name}")
            # Run in background to avoid blocking startup
            asyncio.create_task(preload_model(model_name))
        except Exception as e:
            logger.error(f"Error preloading model {model_name}: {e}")


async def preload_model(model_name: str):
    """Preload a model in the background."""
    try:
        # Load model
        model, tokenizer = load_model(model_name)
        logger.info(f"Model {model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Consyn AI API is running"}


def serve(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    models: Optional[List[str]] = None,
    log_level: str = "info"
):
    """
    Serve the API.
    
    Args:
        host: Host to serve on
        port: Port to serve on
        models: List of models to preload
        log_level: Logging level
    """
    import uvicorn
    
    # Set models to preload
    if models:
        app.state.models_to_preload = models
        
    # Start server
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level=log_level.lower()
    )


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    models_to_preload = os.environ.get("MODELS_TO_PRELOAD", "verse").split(",")
    log_level = os.environ.get("LOG_LEVEL", "info")
    
    # Start server
    serve(
        host=host,
        port=port,
        models=models_to_preload,
        log_level=log_level
    )