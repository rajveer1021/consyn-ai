"""
Consyn AI: A family of language models.
"""

# Import submodules
from . import model
from . import tokenization
from . import training
from . import inference
from . import api

# Export main classes for direct import
from .model import (
    ConsynConfig,
    ConsynVerseConfig,
    ConsynStanzaConfig,
    ConsynEpicConfig,
    ConsynModel,
    ConsynLMHeadModel,
)
from .tokenization import (
    ConsynTokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
)
from .training import (
    ConsynTrainer,
)
from .inference import (
    ConsynInferenceEngine,
)

__version__ = "0.1.0"