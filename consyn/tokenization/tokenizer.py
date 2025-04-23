# consyn/tokenization/tokenizer.py
"""
Base tokenizer interface for Consyn AI models.
This module defines the abstract tokenizer class and common functionality.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import regex as re
import torch
import numpy as np


class ConsynTokenizer(ABC):
    """
    Abstract base class for tokenizers used in Consyn AI models.
    """
    
    def __init__(
        self,
        vocab_file: str = None,
        merges_file: str = None,
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        unk_token: str = "<|endoftext|>",
        add_prefix_space: bool = False,
        **kwargs
    ):
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.add_prefix_space = add_prefix_space
        
        # Special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Load vocabulary and merges
        self.encoder = {}  # token -> id
        self.decoder = {}  # id -> token
        self.bpe_ranks = {}  # merges with their priorities
        
        # Load if files are provided
        if vocab_file is not None:
            self._load_vocab(vocab_file)
            
        if merges_file is not None:
            self._load_merges(merges_file)
            
    @abstractmethod
    def _load_vocab(self, vocab_file: str) -> None:
        """
        Load vocabulary from file.
        
        Args:
            vocab_file: Path to vocabulary file
        """
        pass
        
    @abstractmethod
    def _load_merges(self, merges_file: str) -> None:
        """
        Load BPE merge operations from file.
        
        Args:
            merges_file: Path to merges file
        """
        pass
        
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            list: List of tokens
        """
        pass
        
    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: List of token IDs
        """
        pass
        
    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            list: List of tokens
        """
        pass
        
    def encode(
        self, 
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor, np.ndarray]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: Output format ('pt' for PyTorch, 'np' for NumPy, None for list)
            
        Returns:
            Union[List[int], torch.Tensor, np.ndarray]: Encoded text
        """
        tokens = self.tokenize(text)
        input_ids = [self.vocab.get(t) for t in tokens]
        input_ids = [i for i in input_ids if i is not None]

        if not input_ids:
            print(f"[DEBUG] SKIPPED TEXT: '{text.strip()}' → Tokens: {tokens} → IDs: {input_ids}")

        return input_ids


    def decode(
        self, 
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            str: Decoded text
        """
        # Convert tensors to lists if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
            
        # Handle batched input
        if isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]
            
        # Convert IDs to tokens
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # Filter special tokens if requested
        if skip_special_tokens:
            special_tokens = {self.bos_token, self.eos_token, self.pad_token, self.unk_token}
            tokens = [t for t in tokens if t not in special_tokens]
            
        # Join tokens into text
        text = self._decode_tokens_to_text(tokens)
        
        return text
        
    @abstractmethod
    def _decode_tokens_to_text(self, tokens: List[str]) -> str:
        """
        Join tokens into text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            str: Decoded text
        """
        pass
        
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[Union[str, Tuple[str, str]]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """
        Encode a batch of texts or text pairs.
        
        Args:
            batch_text_or_text_pairs: Batch of texts or text pairs
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length
            return_tensors: Output format ('pt' for PyTorch, 'np' for NumPy, None for list)
            
        Returns:
            dict: Encoded batch with keys 'input_ids', 'attention_mask', etc.
        """
        # Handle batch input
        batch_inputs = []
        
        for text_or_pair in batch_text_or_text_pairs:
            if isinstance(text_or_pair, tuple):
                text, text_pair = text_or_pair
                # For text pairs, encode separately and concatenate
                text_ids = self.encode(text, add_special_tokens=False)
                pair_ids = self.encode(text_pair, add_special_tokens=False)
                
                if add_special_tokens:
                    # Add special tokens: [BOS] text [EOS] text_pair [EOS]
                    bos_id = self.convert_tokens_to_ids([self.bos_token])[0] if self.bos_token else None
                    eos_id = self.convert_tokens_to_ids([self.eos_token])[0] if self.eos_token else None
                    
                    combined_ids = []
                    if bos_id is not None:
                        combined_ids.append(bos_id)
                    combined_ids.extend(text_ids)
                    if eos_id is not None:
                        combined_ids.append(eos_id)
                    combined_ids.extend(pair_ids)
                    if eos_id is not None:
                        combined_ids.append(eos_id)
                else:
                    combined_ids = text_ids + pair_ids
                    
                batch_inputs.append(combined_ids)
            else:
                # For single text, encode normally
                batch_inputs.append(self.encode(text_or_pair, add_special_tokens=add_special_tokens))
                
        # Apply truncation if requested
        if truncation and max_length is not None:
            batch_inputs = [ids[:max_length] for ids in batch_inputs]
            
        # Compute actual max length for padding
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in batch_inputs)
                
            # Pad sequences
            pad_id = self.convert_tokens_to_ids([self.pad_token])[0] if self.pad_token else 0
            batch_inputs = [
                ids + [pad_id] * (max_length - len(ids)) for ids in batch_inputs
            ]
            
        # Create attention masks
        attention_masks = [
            [1] * len(ids) + [0] * (max_length - len(ids)) if padding and max_length > len(ids) else [1] * len(ids)
            for ids in batch_inputs
        ]
        
        # Convert to requested tensor format
        if return_tensors == 'pt':
            batch_inputs = torch.tensor(batch_inputs)
            attention_masks = torch.tensor(attention_masks)
        elif return_tensors == 'np':
            batch_inputs = np.array(batch_inputs)
            attention_masks = np.array(attention_masks)
            
        return {
            'input_ids': batch_inputs,
            'attention_mask': attention_masks,
        }
        
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save tokenizer vocabulary and configuration to directory.
        
        Args:
            save_directory: Directory to save tokenizer files
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(save_directory, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False)
            
        # Save merges if available
        if self.bpe_ranks:
            merges_path = os.path.join(save_directory, 'merges.txt')
            with open(merges_path, 'w', encoding='utf-8') as f:
                for merge, _ in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                    f.write(' '.join(merge) + '\n')
                    
        # Save tokenizer configuration
        config_path = os.path.join(save_directory, 'tokenizer_config.json')
        config = {
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'add_prefix_space': self.add_prefix_space,
            'vocab_file': 'vocab.json',
            'merges_file': 'merges.txt' if self.bpe_ranks else None,
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def from_pretrained(cls, pretrained_dir_or_name: str, **kwargs) -> 'ConsynTokenizer':
        """
        Load a tokenizer from a pretrained directory or name.
        
        Args:
            pretrained_dir_or_name: Directory or name of pretrained tokenizer
            **kwargs: Override config parameters
            
        Returns:
            ConsynTokenizer: Loaded tokenizer
        """
        # Determine if this is a directory or predefined name
        if os.path.isdir(pretrained_dir_or_name):
            # Load from directory
            tokenizer_config_path = os.path.join(pretrained_dir_or_name, 'tokenizer_config.json')
            vocab_path = os.path.join(pretrained_dir_or_name, 'vocab.json')
            merges_path = os.path.join(pretrained_dir_or_name, 'merges.txt')
            
            # Load config
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Override with kwargs
            config.update(kwargs)
            
            config.pop("vocab_file", None)
            config.pop("merges_file", None)

            # Create tokenizer instance
            tokenizer = cls(
                vocab_file=vocab_path if os.path.exists(vocab_path) else None,
                merges_file=merges_path if os.path.exists(merges_path) else None,
                **config
            )
            
            return tokenizer
        else:
            # For predefined names, this would connect to a model hub like Hugging Face
            # For this implementation, we'll raise an error
            raise NotImplementedError(
                f"Loading pretrained tokenizer by name '{pretrained_dir_or_name}' is not implemented."
            )