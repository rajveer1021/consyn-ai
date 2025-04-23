"""
Consyn AI Configuration Management
"""

import os
import json
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Parsed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (verse, stanza, epic)
        
    Returns:
        Model-specific configuration
    """
    config_path = os.path.join(
        os.path.dirname(__file__), 
        'model', 
        f'{model_name}.json'
    )
    return load_config(config_path)

__all__ = [
    'load_config',
    'get_model_config'
]