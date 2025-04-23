"""
Dataset loading and processing for Consyn AI models.
This module provides utilities for loading and preprocessing text datasets for training.
"""

import glob
import json
import os
import random
import logging
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

# Set up logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class TextFileDataset(Dataset):
    """
    Dataset for loading and tokenizing text from a collection of files.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        block_size: int = 1024,
        stride: Optional[int] = None,
        text_preprocessing_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            file_paths: List of paths to text files
            tokenizer: Tokenizer for encoding text
            block_size: Size of text blocks (max sequence length)
            stride: Stride for sliding window. If None, non-overlapping blocks are used.
            text_preprocessing_func: Optional function for text preprocessing
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.text_preprocessing_func = text_preprocessing_func
        
        # Load and tokenize all files
        self.examples = []
        self._load_and_tokenize_files()
        
        logger.info(f"Created TextFileDataset with {len(self.examples)} examples")
        
    def _load_and_tokenize_files(self):
        """Load and tokenize all files in the dataset."""
        total_files = len(self.file_paths)
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Skip empty files
                if not text.strip():
                    logger.warning(f"File {file_path} is empty, skipping")
                    continue
                    
                # Apply preprocessing if provided
                if self.text_preprocessing_func is not None:
                    text = self.text_preprocessing_func(text)
                    
                # Tokenize the entire text
                tokenized_text = self.tokenizer.encode(text, add_special_tokens=False)
                
                # If text is shorter than block_size, add it anyway with padding
                if len(tokenized_text) < self.block_size:
                    logger.warning(f"File {file_path} has only {len(tokenized_text)} tokens, shorter than block_size {self.block_size}")
                    # Pad to block_size
                    padded_tokens = tokenized_text + [self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0] * (self.block_size - len(tokenized_text))
                    self.examples.append(padded_tokens)
                else:
                    # Create blocks of text
                    for i in range(0, len(tokenized_text) - self.block_size + 1, self.stride):
                        self.examples.append(tokenized_text[i:i + self.block_size])
                        
                # Log progress for large datasets
                if file_idx % 100 == 0 and file_idx > 0:
                    logger.info(f"Processed {file_idx}/{total_files} files")
                    
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                
        logger.info(f"Loaded {len(self.examples)} examples from {total_files} files")
        
        # If no examples were loaded, raise a more helpful error
        if len(self.examples) == 0:
            raise ValueError(f"No examples could be extracted from the provided files. "
                            f"Please check that the files contain sufficient text and "
                            f"that the block_size ({self.block_size}) is appropriate.")
                
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
        
    def __getitem__(self, idx):
        """Get a single example from the dataset."""
        # Input IDs are the tokenized text block
        print(f"Example #{idx}: {self.examples[idx]}")
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        
        # For language modeling, labels are identical to input IDs
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for processing large text corpora without loading everything into memory.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        block_size: int = 1024,
        buffer_size: int = 1000,
        shuffle: bool = True,
        text_preprocessing_func: Optional[Callable[[str], str]] = None,
        infinite: bool = False,
    ):
        """
        Initialize the streaming dataset.
        
        Args:
            file_paths: List of paths to text files
            tokenizer: Tokenizer for encoding text
            block_size: Size of text blocks (max sequence length)
            buffer_size: Number of examples to buffer for shuffling
            shuffle: Whether to shuffle examples
            text_preprocessing_func: Optional function for text preprocessing
            infinite: Whether to repeat the dataset infinitely
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.text_preprocessing_func = text_preprocessing_func
        self.infinite = infinite
        
        logger.info(f"Created StreamingTextDataset with {len(file_paths)} files")
        
    def _process_file(self, file_path: str) -> Iterable[List[int]]:
        """
        Process a single file and yield blocks of tokens.
        
        Args:
            file_path: Path to the text file
            
        Yields:
            list: Blocks of token IDs
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Apply preprocessing if provided
            if self.text_preprocessing_func is not None:
                text = self.text_preprocessing_func(text)
                
            # Tokenize the entire text
            tokenized_text = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Create blocks of text
            for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size):
                yield tokenized_text[i:i + self.block_size]
                
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            
    def _get_stream(self) -> Iterable[List[int]]:
        """
        Get a stream of examples from all files.
        
        Yields:
            list: Blocks of token IDs
        """
        file_paths = self.file_paths.copy()
        
        while True:
            # Shuffle files if requested
            if self.shuffle:
                random.shuffle(file_paths)
                
            # Process each file
            for file_path in file_paths:
                yield from self._process_file(file_path)
                
            # Break if not infinite
            if not self.infinite:
                break
                
    def __iter__(self):
        """
        Iterate over the dataset.
        
        Yields:
            dict: Dictionary with input_ids and labels
        """
        # Get the base stream of examples
        stream = self._get_stream()
        
        # Apply buffering and shuffling if requested
        if self.shuffle:
            buffer = []
            for example in stream:
                buffer.append(example)
                
                # Once buffer is full, shuffle and yield examples
                if len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        input_ids = torch.tensor(item, dtype=torch.long)
                        yield {
                            "input_ids": input_ids,
                            "labels": input_ids.clone(),
                        }
                    buffer = []
                    
            # Yield remaining examples
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    input_ids = torch.tensor(item, dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone(),
                    }
        else:
            # No shuffling, directly yield examples
            for example in stream:
                input_ids = torch.tensor(example, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }


class JsonLinesDataset(IterableDataset):
    """
    Dataset for processing JSON Lines files where each line is a JSON object.
    Useful for processing datasets like The Pile.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        block_size: int = 1024,
        buffer_size: int = 1000,
        shuffle: bool = True,
        text_field: str = "text",
        filter_func: Optional[Callable[[Dict], bool]] = None,
        infinite: bool = False,
    ):
        """
        Initialize the JSON Lines dataset.
        
        Args:
            file_paths: List of paths to JSON Lines files
            tokenizer: Tokenizer for encoding text
            block_size: Size of text blocks (max sequence length)
            buffer_size: Number of examples to buffer for shuffling
            shuffle: Whether to shuffle examples
            text_field: Field in JSON objects containing the text
            filter_func: Optional function for filtering JSON objects
            infinite: Whether to repeat the dataset infinitely
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.text_field = text_field
        self.filter_func = filter_func
        self.infinite = infinite
        
        logger.info(f"Created JsonLinesDataset with {len(file_paths)} files")
        
    def _process_file(self, file_path: str) -> Iterable[Dict]:
        """
        Process a single JSON Lines file and yield JSON objects.
        
        Args:
            file_path: Path to the JSON Lines file
            
        Yields:
            dict: JSON objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        json_obj = json.loads(line.strip())
                        
                        # Apply filter if provided
                        if self.filter_func is not None and not self.filter_func(json_obj):
                            continue
                            
                        yield json_obj
                        
                    except json.JSONDecodeError:
                        # Skip invalid JSON
                        logger.warning(f"Invalid JSON at {file_path}:{line_num+1}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            
    def _get_stream(self) -> Iterable[Dict]:
        """
        Get a stream of JSON objects from all files.
        
        Yields:
            dict: JSON objects
        """
        file_paths = self.file_paths.copy()
        
        while True:
            # Shuffle files if requested
            if self.shuffle:
                random.shuffle(file_paths)
                
            # Process each file
            for file_path in file_paths:
                yield from self._process_file(file_path)
                
            # Break if not infinite
            if not self.infinite:
                break
                
    def __iter__(self):
        """
        Iterate over the dataset.
        
        Yields:
            dict: Dictionary with input_ids and labels
        """
        # Get the base stream of JSON objects
        stream = self._get_stream()
        
        # Buffer for examples
        buffer = []
        
        for json_obj in stream:
            # Extract text from JSON object
            if self.text_field in json_obj:
                text = json_obj[self.text_field]
                
                # Tokenize text
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                
                # Create blocks of tokens
                for i in range(0, len(tokens) - self.block_size + 1, self.block_size):
                    block = tokens[i:i + self.block_size]
                    
                    # Skip blocks that are too short
                    if len(block) < self.block_size:
                        # Pad if close to block_size
                        if len(block) > self.block_size * 0.8:
                            block = block + [self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0] * (self.block_size - len(block))
                        else:
                            continue
                            
                    buffer.append(block)
                    
                    # Once buffer is full, shuffle and yield examples
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            input_ids = torch.tensor(item, dtype=torch.long)
                            yield {
                                "input_ids": input_ids,
                                "labels": input_ids.clone(),
                            }
                        buffer = []
                        
        # Yield remaining examples
        if buffer:
            if self.shuffle:
                random.shuffle(buffer)
            for item in buffer:
                input_ids = torch.tensor(item, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }


class ShardedDataset(IterableDataset):
    """
    Dataset for handling pre-sharded data, where each shard is a separate file.
    """
    
    def __init__(
        self,
        shard_pattern: str,
        tokenizer = None,
        buffer_size: int = 1000,
        shuffle: bool = True,
        shuffle_shards: bool = True,
        infinite: bool = False,
        preprocessing_func: Optional[Callable[[Dict], Dict]] = None,
    ):
        """
        Initialize the sharded dataset.
        
        Args:
            shard_pattern: Glob pattern for finding shard files
            tokenizer: Optional tokenizer for decoding/encoding if needed
            buffer_size: Number of examples to buffer for shuffling
            shuffle: Whether to shuffle examples
            shuffle_shards: Whether to shuffle the order of shards
            infinite: Whether to repeat the dataset infinitely
            preprocessing_func: Optional function for preprocessing examples
        """
        self.shard_paths = glob.glob(shard_pattern)
        if not self.shard_paths:
            raise ValueError(f"No files found matching pattern: {shard_pattern}")
            
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.shuffle_shards = shuffle_shards
        self.infinite = infinite
        self.preprocessing_func = preprocessing_func
        
        logger.info(f"Created ShardedDataset with {len(self.shard_paths)} shards")
        
    def _process_shard(self, shard_path: str) -> Iterable[Dict]:
        """
        Process a single shard and yield examples.
        
        Args:
            shard_path: Path to the shard file
            
        Yields:
            dict: Examples from the shard
        """
        try:
            # Determine file format based on extension
            ext = os.path.splitext(shard_path)[1].lower()
            
            if ext == '.jsonl':
                # Process as JSON Lines
                with open(shard_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        try:
                            example = json.loads(line.strip())
                            
                            # Apply preprocessing if provided
                            if self.preprocessing_func is not None:
                                example = self.preprocessing_func(example)
                                
                            yield example
                            
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            logger.warning(f"Invalid JSON at {shard_path}:{line_num+1}")
                            continue
                            
            elif ext in ['.npy', '.npz']:
                # Process as NumPy array
                try:
                    if ext == '.npy':
                        data = np.load(shard_path, allow_pickle=True)
                        if isinstance(data, np.ndarray) and data.dtype == np.dtype('O'):
                            # Array of objects
                            for example in data:
                                # Apply preprocessing if provided
                                if self.preprocessing_func is not None:
                                    example = self.preprocessing_func(example)
                                    
                                yield example
                        else:
                            # Single array
                            yield data
                    else:  # .npz
                        data = np.load(shard_path)
                        for key in data.keys():
                            yield {key: data[key]}
                except Exception as e:
                    logger.warning(f"Error loading NumPy file {shard_path}: {e}")
                    
            elif ext in ['.pt', '.pth']:
                # Process as PyTorch tensor
                try:
                    data = torch.load(shard_path, map_location="cpu")
                    
                    if isinstance(data, dict):
                        # Single example
                        # Apply preprocessing if provided
                        if self.preprocessing_func is not None:
                            data = self.preprocessing_func(data)
                            
                        yield data
                    elif isinstance(data, list):
                        # List of examples
                        for example in data:
                            # Apply preprocessing if provided
                            if self.preprocessing_func is not None:
                                example = self.preprocessing_func(example)
                                
                            yield example
                    else:
                        # Unknown format
                        yield {"data": data}
                except Exception as e:
                    logger.warning(f"Error loading PyTorch file {shard_path}: {e}")
                    
            else:
                # Unknown format, try to parse as text
                try:
                    with open(shard_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # Apply preprocessing if provided
                    if self.preprocessing_func is not None:
                        text = self.preprocessing_func(text)
                        
                    if self.tokenizer is not None:
                        tokens = self.tokenizer.encode(text)
                        yield {
                            "input_ids": tokens,
                            "labels": tokens,
                        }
                    else:
                        yield {
                            "text": text,
                        }
                except Exception as e:
                    logger.warning(f"Error processing file {shard_path} as text: {e}")
                    
        except Exception as e:
            logger.warning(f"Error processing shard {shard_path}: {e}")
            
    def _get_stream(self) -> Iterable[Dict]:
        """
        Get a stream of examples from all shards.
        
        Yields:
            dict: Examples
        """
        shard_paths = self.shard_paths.copy()
        
        while True:
            # Shuffle shards if requested
            if self.shuffle_shards:
                random.shuffle(shard_paths)
                
            # Process each shard
            for shard_path in shard_paths:
                yield from self._process_shard(shard_path)
                
            # Break if not infinite
            if not self.infinite:
                break
                
    def __iter__(self):
        """
        Iterate over the dataset.
        
        Yields:
            dict: Examples
        """
        # Get the base stream of examples
        stream = self._get_stream()
        
        # Apply buffering and shuffling if requested
        if self.shuffle:
            buffer = []
            for example in stream:
                buffer.append(example)
                
                # Once buffer is full, shuffle and yield examples
                if len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []
                    
            # Yield remaining examples
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    yield item
        else:
            # No shuffling, directly yield examples
            for example in stream:
                yield example


def get_dataset(
    data_path: Union[str, List[str]],
    tokenizer,
    dataset_type: str = "text",
    block_size: int = 1024,
    streaming: bool = True,
    buffer_size: int = 1000,
    shuffle: bool = True,
    **kwargs
) -> Union[Dataset, IterableDataset]:
    """
    Factory function to create appropriate dataset based on data type.
    
    Args:
        data_path: Path to data (file, directory, or pattern) or list of paths
        tokenizer: Tokenizer for encoding text
        dataset_type: Type of dataset ('text', 'jsonl', 'sharded', or 'huggingface')
        block_size: Size of text blocks (max sequence length)
        streaming: Whether to use streaming dataset for large data
        buffer_size: Buffer size for streaming datasets
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for specific dataset types
        
    Returns:
        Dataset or IterableDataset: Dataset for training
    """
    # Handle single path or list of paths
    if isinstance(data_path, str):
        if os.path.isdir(data_path):
            # If directory, get all files
            file_paths = glob.glob(os.path.join(data_path, "**/*.*"), recursive=True)
        elif "*" in data_path:
            # If pattern, use glob
            file_paths = glob.glob(data_path)
        else:
            # Single file
            file_paths = [data_path]
    else:
        # List of paths
        file_paths = data_path
        
    if not file_paths:
        raise ValueError(f"No files found at {data_path}")
        
    logger.info(f"Found {len(file_paths)} files for dataset")
    
    # Create dataset based on type
    if dataset_type == "text":
        if streaming:
            return StreamingTextDataset(
                file_paths=file_paths,
                tokenizer=tokenizer,
                block_size=block_size,
                buffer_size=buffer_size,
                shuffle=shuffle,
                **kwargs
            )
        else:
            return TextFileDataset(
                file_paths=file_paths,
                tokenizer=tokenizer,
                block_size=block_size,
                **kwargs
            )
    elif dataset_type == "jsonl":
        return JsonLinesDataset(
            file_paths=file_paths,
            tokenizer=tokenizer,
            block_size=block_size,
            buffer_size=buffer_size,
            shuffle=shuffle,
            **kwargs
        )
    elif dataset_type == "sharded":
        return ShardedDataset(
            shard_pattern=data_path if isinstance(data_path, str) else file_paths[0],
            tokenizer=tokenizer,
            buffer_size=buffer_size,
            shuffle=shuffle,
            **kwargs
        )
    elif dataset_type == "huggingface":
        if not HAS_DATASETS:
            raise ImportError("The 'datasets' library is required for HuggingFace datasets.")
            
        # Load dataset from HuggingFace
        dataset_name = kwargs.get("dataset_name", data_path)
        dataset_config = kwargs.get("dataset_config")
        dataset_split = kwargs.get("dataset_split", "train")
        text_field = kwargs.get("text_field", "text")
        
        # Load the dataset
        if streaming:
            ds = datasets.load_dataset(
                dataset_name,
                dataset_config,
                split=dataset_split,
                streaming=True,
            )
            
            # Apply preprocessing and tokenization
            def preprocess_function(examples):
                texts = examples[text_field]
                tokenized = tokenizer.encode(texts, add_special_tokens=True)
                
                # Truncate or pad
                if len(tokenized) > block_size:
                    tokenized = tokenized[:block_size]
                
                input_ids = torch.tensor(tokenized, dtype=torch.long)
                return {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }
                
            return ds.map(preprocess_function)
        else:
            ds = datasets.load_dataset(
                dataset_name,
                dataset_config,
                split=dataset_split,
            )
            
            # Apply preprocessing and tokenization
            def preprocess_function(examples):
                texts = examples[text_field]
                result = tokenizer.batch_encode_plus(
                    texts,
                    max_length=block_size,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                result["labels"] = result["input_ids"].clone()
                return result
                
            return ds.map(
                preprocess_function,
                batched=True,
                remove_columns=ds.column_names,
            )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data (ignored for IterableDataset)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in GPU
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    # Determine if dataset is map-style or iterable-style
    is_iterable = isinstance(dataset, IterableDataset)
    
    # Warn if trying to shuffle an IterableDataset
    if shuffle and is_iterable:
        logger.warning("shuffle=True has no effect for IterableDataset. The dataset should handle shuffling internally.")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not is_iterable,  # Shuffle only if not IterableDataset
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batches for efficient training
    )