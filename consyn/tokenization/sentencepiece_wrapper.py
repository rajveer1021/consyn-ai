# consyn/tokenization/sentencepiece_wrapper.py
"""
SentencePiece tokenizer wrapper for Consyn AI models.
This module provides a wrapper around the SentencePiece tokenizer.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

try:
    import sentencepiece as spm
except ImportError:
    spm = None

from .tokenizer import ConsynTokenizer


class SentencePieceTokenizer(ConsynTokenizer):
    """
    Wrapper around SentencePiece tokenizer for Consyn AI models.
    """
    
    def __init__(
        self,
        model_file: str = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        add_prefix_space: bool = False,
        **kwargs
    ):
        # Check if sentencepiece is installed
        if spm is None:
            raise ImportError("SentencePiece is not installed. Install it with: pip install sentencepiece")
            
        self.model_file = model_file
        
        # Load SentencePiece model if provided
        if model_file is not None and os.path.exists(model_file):
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(model_file)
        else:
            self.sp_model = None
            
        # Call parent constructor
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            **kwargs
        )
        
        # Build encoder/decoder mappings from SentencePiece
        if self.sp_model is not None:
            self._build_encodings_from_sentencepiece()
            
    def _build_encodings_from_sentencepiece(self) -> None:
        """
        Build token-to-id and id-to-token mappings from SentencePiece model.
        """
        self.encoder = {}
        self.decoder = {}
        
        for i in range(self.sp_model.GetPieceSize()):
            piece = self.sp_model.IdToPiece(i)
            self.encoder[piece] = i
            self.decoder[i] = piece
            
    def _load_vocab(self, vocab_file: str) -> None:
        """
        Not used for SentencePiece tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file
        """
        # SentencePiece includes its own vocabulary
        pass
        
    def _load_merges(self, merges_file: str) -> None:
        """
        Not used for SentencePiece tokenizer.
        
        Args:
            merges_file: Path to merges file
        """
        # SentencePiece includes its own merge operations
        pass
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            list: List of tokens
        """
        if self.sp_model is None:
            raise ValueError("SentencePiece model is not loaded.")
            
        # Add prefix space if specified
        if self.add_prefix_space and not text.startswith(' '):
            text = ' ' + text
            
        # Tokenize with SentencePiece
        tokens = self.sp_model.EncodeAsPieces(text)
        
        return tokens
        
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: List of token IDs
        """
        if self.sp_model is None:
            raise ValueError("SentencePiece model is not loaded.")
            
        # Use SentencePiece to convert tokens to IDs
        ids = [self.sp_model.PieceToId(token) for token in tokens]
        
        return ids
        
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            list: List of tokens
        """
        if self.sp_model is None:
            raise ValueError("SentencePiece model is not loaded.")
            
        # Use SentencePiece to convert IDs to tokens
        tokens = [self.sp_model.IdToPiece(id) for id in ids]
        
        return tokens
        
    def _decode_tokens_to_text(self, tokens: List[str]) -> str:
        """
        Join tokens into text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            str: Decoded text
        """
        if self.sp_model is None:
            raise ValueError("SentencePiece model is not loaded.")
            
        # Use SentencePiece to decode tokens to text
        text = self.sp_model.DecodePieces(tokens)
        
        return text
        
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save tokenizer model and configuration to directory.
        
        Args:
            save_directory: Directory to save tokenizer files
        """
        if self.sp_model is None:
            raise ValueError("SentencePiece model is not loaded.")
            
        os.makedirs(save_directory, exist_ok=True)
        
        # Save SentencePiece model
        model_path = os.path.join(save_directory, 'spiece.model')
        self.sp_model.Save(model_path)
        
        # Save tokenizer configuration
        super().save_pretrained(save_directory)
        
    @classmethod
    def train(
        cls,
        files: List[str],
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
        model_type: str = "unigram",  # or "bpe", "char", "word"
        special_tokens: List[str] = ["<s>", "</s>", "<pad>", "<unk>"],
        output_dir: str = "./tokenizer",
    ) -> 'SentencePieceTokenizer':
        """
        Train a new SentencePiece tokenizer.
        
        Args:
            files: List of files to train on
            vocab_size: Size of the vocabulary
            character_coverage: Character coverage
            model_type: SentencePiece model type
            special_tokens: List of special tokens
            output_dir: Directory to save tokenizer files
            
        Returns:
            SentencePieceTokenizer: Trained tokenizer
        """
        # Check if sentencepiece is installed
        if spm is None:
            raise ImportError("SentencePiece is not installed. Install it with: pip install sentencepiece")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare input text file list
        input_file_list = ','.join(files)
        model_prefix = os.path.join(output_dir, 'spiece')
        
        # Prepare training arguments
        user_defined_symbols = ','.join(special_tokens)
        
        # Train the model
        spm.SentencePieceTrainer.Train(
            f'--input={input_file_list} '
            f'--model_prefix={model_prefix} '
            f'--vocab_size={vocab_size} '
            f'--character_coverage={character_coverage} '
            f'--model_type={model_type} '
            f'--user_defined_symbols={user_defined_symbols} '
            f'--unk_id=3 '
            f'--bos_id=0 '
            f'--eos_id=1 '
            f'--pad_id=2 '
        )
        
        # Load the trained model
        tokenizer = cls(
            model_file=f'{model_prefix}.model',
            bos_token=special_tokens[0] if len(special_tokens) > 0 else "<s>",
            eos_token=special_tokens[1] if len(special_tokens) > 1 else "</s>",
            pad_token=special_tokens[2] if len(special_tokens) > 2 else "<pad>",
            unk_token=special_tokens[3] if len(special_tokens) > 3 else "<unk>",
        )
        
        return tokenizer