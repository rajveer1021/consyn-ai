# consyn/api/models.py
"""
Pydantic models for Consyn AI API.
This module defines the request and response models for the API.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(128, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_k: int = Field(50, description="Top-K filtering value")
    top_p: float = Field(0.9, description="Top-P filtering value")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    num_return_sequences: int = Field(1, description="Number of sequences to return")
    model_name: str = Field("verse", description="Model name (verse, stanza, or epic)")
    
    @validator("max_new_tokens")
    def validate_max_tokens(cls, v):
        if v < 1 or v > 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        return v
        
    @validator("temperature")
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v
        
    @validator("model_name")
    def validate_model_name(cls, v):
        valid_models = ["verse", "stanza", "epic"]
        if v.lower() not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")
        return v.lower()


class StreamGenerationRequest(GenerationRequest):
    """Request model for streaming text generation."""
    
    stream: bool = Field(True, description="Whether to stream the response")


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""
    
    prompts: List[str] = Field(..., description="List of input prompts for generation")
    max_new_tokens: int = Field(128, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_k: int = Field(50, description="Top-K filtering value")
    top_p: float = Field(0.9, description="Top-P filtering value")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    batch_size: int = Field(4, description="Batch size for generation")
    model_name: str = Field("verse", description="Model name (verse, stanza, or epic)")
    
    @validator("batch_size")
    def validate_batch_size(cls, v):
        if v < 1 or v > 64:
            raise ValueError("batch_size must be between 1 and 64")
        return v
        
    @validator("model_name")
    def validate_model_name(cls, v):
        valid_models = ["verse", "stanza", "epic"]
        if v.lower() not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")
        return v.lower()
    
    @validator("max_new_tokens")
    def validate_max_tokens(cls, v):
        if v < 1 or v > 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        return v
        
    @validator("temperature")
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str = Field(..., description="Generated text")
    generation_time: float = Field(..., description="Generation time in seconds")


class BatchGenerationResponse(BaseModel):
    """Response model for batch text generation."""
    
    generated_texts: List[str] = Field(..., description="List of generated texts")
    generation_time: float = Field(..., description="Generation time in seconds")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    parameters: int = Field(..., description="Number of parameters")
    max_context_length: int = Field(..., description="Maximum context length")
    vocab_size: int = Field(..., description="Vocabulary size")
    loaded: bool = Field(..., description="Whether the model is loaded")


class StreamingChunkResponse(BaseModel):
    """Response chunk for streaming generation."""
    
    token: str = Field(..., description="Generated token")
    index: int = Field(..., description="Token index")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing generation")
    full_text: Optional[str] = Field(None, description="Full generated text (only in final chunk)")