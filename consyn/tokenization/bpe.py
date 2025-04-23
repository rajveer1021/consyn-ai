# consyn/tokenization/bpe.py
"""
Byte-Pair Encoding (BPE) tokenizer implementation for Consyn AI models.
This module provides a BPE tokenizer similar to the one used in GPT-2.
"""

import json
import os
import regex as re
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import numpy as np

from .tokenizer import ConsynTokenizer


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    
    This is used to avoid issues with tokenization of non-ASCII characters
    and allows handling of byte-level tokens as string tokens.
    
    Based on GPT-2 tokenizer implementation.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: List[str]) -> Set[Tuple[str, str]]:
    """
    Return all bigram pairs in a word.
    
    Args:
        word: List of characters/tokens
        
    Returns:
        set: Set of character/token pairs
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPETokenizer(ConsynTokenizer):
    """
    Byte-Pair Encoding tokenizer for Consyn AI models.
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
        # Initialize with byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Pattern for tokenization
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if vocab_file:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {}
            
        if merges_file:
            with open(merges_file, "r", encoding="utf-8") as f:
                self.merges = f.readlines()
        else:
            self.merges = []

        # Call parent constructor
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            **kwargs
        )
        
    def _load_vocab(self, vocab_file: str) -> None:
        """
        Load vocabulary from file.
        
        Args:
            vocab_file: Path to vocabulary file (JSON format)
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
            
        self.decoder = {v: k for k, v in self.encoder.items()}
        
    def _load_merges(self, merges_file: str) -> None:
        """
        Load BPE merge operations from file.
        
        Args:
            merges_file: Path to merges file (text format)
        """
        with open(merges_file, 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')
            
        # Filter empty lines
        merges = [tuple(merge.split()) for merge in merges if merge]
        
        # Create dictionary of merge priorities
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
    def bpe(self, token: str) -> str:
        """
        Apply BPE encoding to a token.
        
        Args:
            token: Input token (as string)
            
        Returns:
            str: BPE-encoded token
        """
        if not self.bpe_ranks:
            return token
            
        # Split token into characters
        word = list(token)
        
        # Get all pairs in the word
        pairs = get_pairs(word)
        
        if not pairs:
            return token
            
        while True:
            # Find the highest priority pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # If the pair is not in the merges list, stop
            if bigram not in self.bpe_ranks:
                break
                
            # Split the bigram
            first, second = bigram
            
            # Create new word with merged tokens
            new_word = []
            i = 0
            while i < len(word):
                # Find occurence of the first token
                try:
                    j = word.index(first, i)
                    # Add all tokens before the first token
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # No more occurrences
                    new_word.extend(word[i:])
                    break
                    
                # Check if the second token follows
                if (i < len(word) - 1 and word[i] == first and word[i + 1] == second):
                    # Merge the tokens
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            # Update the word
            word = new_word
            
            # If the word has only one token left, stop
            if len(word) == 1:
                break
                
            # Update pairs
            pairs = get_pairs(word)
            
        # Join the tokens
        return ' '.join(word)
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            list: List of tokens
        """
        # Add prefix space if specified
        if self.add_prefix_space and not text.startswith(' '):
            text = ' ' + text
            
        # Split text using regex pattern
        matches = re.findall(self.pat, text)
        
        # Encode each match as a list of bytes, then convert to unicode representation
        tokens = []
        for match in matches:
            # Convert to bytes and then to unicode chars
            token = ''.join(self.byte_encoder[b] for b in match.encode('utf-8'))
            
            # Apply BPE
            tokens.extend(self.bpe(token).split(' '))
            
        return tokens
        
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: List of token IDs
        """
        # Use the vocabulary to convert tokens to IDs
        ids = []
        for token in tokens:
            # Use UNK ID for unknown tokens
            id = self.encoder.get(token, self.encoder.get(self.unk_token))
            ids.append(id)
            
        return ids
        
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Args:
            ids: List of token IDs
            
        Returns:
            list: List of tokens
        """
        # Use the vocabulary to convert IDs to tokens
        tokens = []
        for id in ids:
            # Use UNK token for unknown IDs
            token = self.decoder.get(id, self.unk_token)
            tokens.append(token)
            
        return tokens
        
    def _decode_tokens_to_text(self, tokens: List[str]) -> str:
        """
        Join tokens into text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            str: Decoded text
        """
        # Join tokens into text
        text = ''.join(tokens)
        
        # Decode byte sequences
        text_bytes = bytearray([self.byte_decoder[c] for c in text])
        
        # Decode as UTF-8
        text = text_bytes.decode('utf-8', errors='replace')
        
        return text
        
    @classmethod
    def train(
        cls,
        files: List[str],
        vocab_size: int = 50257,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<|endoftext|>"],
        output_dir: str = "./tokenizer",
    ) -> 'BPETokenizer':
        """
        Train a new BPE tokenizer from scratch.
        
        Args:
            files: List of files to train on
            vocab_size: Size of the vocabulary to create
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens to add to the vocabulary
            output_dir: Directory to save tokenizer files
            
        Returns:
            BPETokenizer: Trained tokenizer
        """
        # This implementation would be quite complex
        # For a production system, you might want to use an existing implementation like tokenizers
        # Here we'll provide a simplified outline of the process
        
        # 1. Collect initial vocabulary of character-level tokens
        # 2. Learn merges by iteratively merging the most frequent pairs
        # 3. Create vocabulary and merges files
        # 4. Create and return tokenizer
        
        # For now, just raise a NotImplementedError
        raise NotImplementedError(
            "Training a BPE tokenizer from scratch is not implemented in this example. "
            "In a production system, consider using libraries like 'tokenizers'."
        )
