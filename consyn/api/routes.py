# consyn/api/routes.py
"""
API routes for Consyn AI models.
This module defines the HTTP routes for text generation and model information.
"""

import os
import time
import json
import logging
import torch
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from starlette.responses import StreamingResponse

from .models import (
    GenerationRequest,
    StreamGenerationRequest,
    BatchGenerationRequest,
    GenerationResponse,
    BatchGenerationResponse,
    ModelInfoResponse,
)
from .utils import get_inference_engine, load_model, get_model_path, model_cache

# Create router
router = APIRouter()


@router.get("/", summary="Root endpoint")
async def root():
    """Root endpoint."""
    return {"message": "Consyn AI API is running"}


@router.get("/models", response_model=List[ModelInfoResponse], summary="List available models")
async def list_models():
    """List available models."""
    models = []
    
    # Add default models
    for model_name in ["verse", "stanza", "epic"]:
        # Get model path
        model_path = get_model_path(model_name)
        
        # Check if model exists
        loaded = model_name in model_cache
        exists = os.path.exists(model_path)
        
        # Get config
        if loaded:
            model = model_cache[model_name]
            config = model.config
            
            parameters = 0
            for param in model.parameters():
                parameters += param.numel()
                
            max_context_length = config.max_position_embeddings
            vocab_size = config.vocab_size
        elif exists:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    
                parameters = config_dict.get("parameters", 0)
                max_context_length = config_dict.get("max_position_embeddings", 2048)
                vocab_size = config_dict.get("vocab_size", 50257)
            else:
                # Use defaults based on model name
                if model_name == "verse":
                    parameters = 125_000_000
                    max_context_length = 1024
                    vocab_size = 50257
                elif model_name == "stanza":
                    parameters = 1_300_000_000
                    max_context_length = 2048
                    vocab_size = 50257
                elif model_name == "epic":
                    parameters = 13_000_000_000
                    max_context_length = 4096
                    vocab_size = 50257
        else:
            # Use defaults based on model name
            if model_name == "verse":
                parameters = 125_000_000
                max_context_length = 1024
                vocab_size = 50257
            elif model_name == "stanza":
                parameters = 1_300_000_000
                max_context_length = 2048
                vocab_size = 50257
            elif model_name == "epic":
                parameters = 13_000_000_000
                max_context_length = 4096
                vocab_size = 50257
                
        models.append(
            ModelInfoResponse(
                model_name=model_name,
                model_type=f"Consyn {model_name.title()}",
                parameters=parameters,
                max_context_length=max_context_length,
                vocab_size=vocab_size,
                loaded=loaded,
            )
        )
        
    return models


@router.post("/generate", response_model=GenerationResponse, summary="Generate text")
async def generate(request: GenerationRequest):
    """
    Generate text based on a prompt.
    
    Args:
        request: Generation request
        
    Returns:
        GenerationResponse: Generation response
    """
    # Get inference engine
    engine = get_inference_engine(request.model_name)
    
    # Generate text
    start_time = time.time()
    
    generated_texts = engine.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
        num_return_sequences=request.num_return_sequences,
    )
    
    generation_time = time.time() - start_time
    
    # Return first generated text
    return GenerationResponse(
        generated_text=generated_texts[0] if isinstance(generated_texts, list) else generated_texts,
        generation_time=generation_time,
    )


@router.post("/generate/batch", response_model=BatchGenerationResponse, summary="Generate text in batch")
async def generate_batch(request: BatchGenerationRequest):
    """
    Generate text for multiple prompts in batch.
    
    Args:
        request: Batch generation request
        
    Returns:
        BatchGenerationResponse: Batch generation response
    """
    # Get inference engine
    engine = get_inference_engine(request.model_name)
    
    # Generate text
    start_time = time.time()
    
    generated_texts = engine.batch_generate(
        prompts=request.prompts,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
        batch_size=request.batch_size,
    )
    
    generation_time = time.time() - start_time
    
    return BatchGenerationResponse(
        generated_texts=generated_texts,
        generation_time=generation_time,
    )


@router.post("/generate/stream", summary="Stream text generation")
async def generate_stream(request: StreamGenerationRequest):
    """
    Generate text with streaming response.
    
    Args:
        request: Stream generation request
        
    Returns:
        StreamingResponse: Streaming response
    """
    # Get inference engine
    engine = get_inference_engine(request.model_name)
    
    # Define callback for streaming
    async def stream_generator():
        token_buffer = []
        token_index = 0
        
        def token_callback(token, index, probability):
            nonlocal token_buffer, token_index
            token_buffer.append(token)
            token_index = index
            
            # Create response
            response = {
                "token": token,
                "index": index,
                "finish_reason": None,
            }
            
            # Convert to JSON and yield
            return json.dumps(response) + "\n"
            
        # Generate text with streaming
        full_text = engine.generate_stream(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            callback=token_callback,
        )
        
        # Final response
        final_response = {
            "token": "",
            "index": token_index + 1,
            "finish_reason": "stop",
            "full_text": full_text,
        }
        
        # Convert to JSON and yield
        yield json.dumps(final_response) + "\n"
        
    # Return streaming response
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
    )


@router.get("/health", summary="Health check")
async def health_check():
    """Health check endpoint."""
    # Check if at least one model is loaded
    models_loaded = len(model_cache) > 0
    
    # Check if any GPUs are available
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
    }