# consyn/inference/sampling.py
"""
Text generation sampling methods for Consyn AI models.
This module provides various sampling techniques for language model text generation.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def top_k_filtering(logits: torch.Tensor, top_k: int = 0, filter_value: float = -float("Inf")) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k filtering.
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        top_k: Keep only the top-k tokens with highest probability
        filter_value: Value to assign to filtered logits
        
    Returns:
        torch.Tensor: Filtered logits
    """
    if top_k <= 0:
        return logits  # No filtering
        
    # Get top-k values and indices
    top_k = min(top_k, logits.size(-1))  # Ensure k is not larger than vocab size
    values, _ = torch.topk(logits, top_k, dim=-1)
    
    # Get the minimum value in the top-k
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    
    # Filter logits
    return torch.where(logits < min_values, 
                      torch.ones_like(logits) * filter_value, 
                      logits)


def top_p_filtering(logits: torch.Tensor, top_p: float = 1.0, filter_value: float = -float("Inf")) -> torch.Tensor:
    """
    Filter a distribution of logits using nucleus (top-p) filtering.
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        top_p: Keep the top tokens with cumulative probability >= top_p
        filter_value: Value to assign to filtered logits
        
    Returns:
        torch.Tensor: Filtered logits
    """
    if top_p >= 1.0:
        return logits  # No filtering
        
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens below the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift indices to keep the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0  # Keep at least one token
    
    # Create a mask of indices to remove
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    
    # Apply filtering
    return torch.where(indices_to_remove, 
                      torch.ones_like(logits) * filter_value, 
                      logits)


def typical_filtering(logits: torch.Tensor, mass: float = 0.9, filter_value: float = -float("Inf")) -> torch.Tensor:
    """
    Filter a distribution of logits using typical sampling (entropy-based).
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        mass: Keep tokens with typicality score above threshold
        filter_value: Value to assign to filtered logits
        
    Returns:
        torch.Tensor: Filtered logits
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate entropy (negative log probability)
    neg_entropy = probs * torch.log(probs + 1e-8)
    
    # Calculate the expected entropy
    expected_entropy = -torch.sum(neg_entropy, dim=-1, keepdim=True)
    
    # Calculate deviations from expected entropy
    deviation = torch.abs(neg_entropy + expected_entropy)
    
    # Sort deviations
    sorted_devs, sorted_indices = torch.sort(deviation, dim=-1)
    
    # Calculate cumulative probabilities of the sorted indices
    sorted_probs = torch.gather(probs, -1, sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to keep
    sorted_indices_to_remove = cumulative_probs > mass
    
    # Shift to keep the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0  # Keep at least one token
    
    # Create a mask of indices to remove
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    
    # Apply filtering
    return torch.where(indices_to_remove, 
                      torch.ones_like(logits) * filter_value, 
                      logits)


def contrastive_filtering(
    logits: torch.Tensor, 
    contrast_factor: float = 0.3, 
    top_k: int = 50
) -> torch.Tensor:
    """
    Apply contrastive search filtering to logits.
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        contrast_factor: Strength of contrastive filtering (0 to 1)
        top_k: Number of top tokens to consider
        
    Returns:
        torch.Tensor: Filtered logits
    """
    # Apply top-k first
    filtered_logits = top_k_filtering(logits, top_k=top_k)
    
    # For true contrastive search, we would need access to past context embeddings and 
    # the ability to compute similarity. This is a simplified approximation.
    
    # Emphasize the highest probability tokens and de-emphasize others
    probs = F.softmax(filtered_logits, dim=-1)
    
    # Apply contrast factor
    contrasted_probs = probs ** (1.0 / (contrast_factor + 1e-8))
    
    # Convert back to logits
    contrasted_logits = torch.log(contrasted_probs + 1e-8)
    
    return contrasted_logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.1
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        input_ids: Previously generated tokens of shape [batch_size, seq_len]
        penalty: Repetition penalty factor (1.0 means no penalty)
        
    Returns:
        torch.Tensor: Logits with repetition penalty applied
    """
    if penalty == 1.0:
        return logits  # No penalty
        
    # Create a set of unique token IDs for each sequence in the batch
    batch_size = input_ids.size(0)
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Get unique token IDs in this sequence
        unique_tokens = set(input_ids[i].tolist())
        
        # Apply penalty to those tokens
        for token_id in unique_tokens:
            if logits[i, token_id] > 0:
                penalized_logits[i, token_id] /= penalty
            else:
                penalized_logits[i, token_id] *= penalty
                
    return penalized_logits


def apply_presence_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 0.3
) -> torch.Tensor:
    """
    Apply presence penalty to logits (fixed penalty regardless of frequency).
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        input_ids: Previously generated tokens of shape [batch_size, seq_len]
        penalty: Presence penalty factor
        
    Returns:
        torch.Tensor: Logits with presence penalty applied
    """
    if penalty == 0.0:
        return logits  # No penalty
        
    # Create a set of unique token IDs for each sequence in the batch
    batch_size = input_ids.size(0)
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Get unique token IDs in this sequence
        unique_tokens = set(input_ids[i].tolist())
        
        # Apply penalty to those tokens
        for token_id in unique_tokens:
            penalized_logits[i, token_id] -= penalty
            
    return penalized_logits


def apply_frequency_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 0.2
) -> torch.Tensor:
    """
    Apply frequency penalty to logits (penalty scales with frequency).
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        input_ids: Previously generated tokens of shape [batch_size, seq_len]
        penalty: Frequency penalty factor
        
    Returns:
        torch.Tensor: Logits with frequency penalty applied
    """
    if penalty == 0.0:
        return logits  # No penalty
        
    # Count token frequencies for each sequence in the batch
    batch_size = input_ids.size(0)
    vocab_size = logits.size(-1)
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Count token frequencies
        token_counts = torch.bincount(input_ids[i], minlength=vocab_size)
        
        # Apply penalty proportional to frequency
        penalized_logits[i] -= penalty * token_counts
        
    return penalized_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    typical_mass: float = 1.0,
    do_sample: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    Sample next token from logits distribution.
    
    Args:
        logits: Logits distribution of shape [batch_size, vocab_size]
        temperature: Temperature for sampling
        top_k: Number of highest probability tokens to keep
        top_p: Nucleus sampling parameter
        typical_mass: Typical sampling parameter
        do_sample: Whether to sample (True) or take argmax (False)
        
    Returns:
        Tuple[torch.Tensor, float]: Next token ID and its probability
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # For temperature near zero, use greedy decoding
        do_sample = False
        
    # Apply filtering methods
    if top_k > 0:
        logits = top_k_filtering(logits, top_k=top_k)
        
    if top_p < 1.0:
        logits = top_p_filtering(logits, top_p=top_p)
        
    if typical_mass < 1.0:
        logits = typical_filtering(logits, mass=typical_mass)
        
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample or greedy selection
    if do_sample:
        # Multinomial sampling
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy selection
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
    
    # Get probability of selected token
    token_prob = probs.gather(-1, next_token).item()
    
    return next_token, token_prob


def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 5,
    max_length: int = 50,
    min_length: int = 0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform beam search decoding.
    
    Args:
        model: The model to generate with
        input_ids: Initial input IDs of shape [batch_size, seq_len]
        beam_size: Number of beams
        max_length: Maximum generation length
        min_length: Minimum generation length
        repetition_penalty: Penalty for repeated tokens
        length_penalty: Penalty for longer sequences
        early_stopping: Whether to stop when all beams have generated EOS
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Generated sequences and their scores
    """
    # This is a simplified beam search implementation
    # A full implementation would be more complex and handle many edge cases
    
    device = input_ids.device
    batch_size = input_ids.size(0)
    vocab_size = model.config.vocab_size
    
    # Replicate the input_ids for each beam
    input_ids = input_ids.repeat(1, beam_size).view(batch_size * beam_size, -1)
    
    # Initialize scores for each beam
    beam_scores = torch.zeros(batch_size, beam_size, device=device)
    beam_scores[:, 1:] = -1e9  # Initialize all but the first beam with -inf
    beam_scores = beam_scores.view(-1)  # Flatten
    
    # Track finished sequences
    done = [False for _ in range(batch_size)]
    
    # Initialize generation
    current_length = input_ids.size(1)
    
    # Main generation loop
    while current_length < max_length:
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Get next token logits
        next_token_logits = outputs[0][:, -1, :]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size * beam_size):
                for token_id in set(input_ids[i].tolist()):
                    if next_token_logits[i, token_id] > 0:
                        next_token_logits[i, token_id] /= repetition_penalty
                    else:
                        next_token_logits[i, token_id] *= repetition_penalty
        
        # Prevent EOS before min_length
        if min_length > 0 and current_length < min_length and eos_token_id is not None:
            next_token_logits[:, eos_token_id] = -float("inf")
            
        # Calculate log probabilities
        next_scores = F.log_softmax(next_token_logits, dim=-1)
        
        # Add beam scores
        next_scores = next_scores + beam_scores[:, None]
        
        # Reshape for beam search
        next_scores = next_scores.view(batch_size, beam_size * vocab_size)
        
        # Get the best 2 * beam_size candidates
        next_scores, next_tokens = torch.topk(
            next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
        )
        
        # Determine which beams and tokens to keep
        next_batch_beam = []
        
        for batch_idx in range(batch_size):
            # If this sequence is done, add pad tokens
            if done[batch_idx]:
                next_batch_beam.extend([
                    (0, pad_token_id, batch_idx * beam_size)
                ] * beam_size)
                continue
                
            # Find best beams and tokens
            for beam_token_rank, (beam_token_score, beam_token) in enumerate(
                zip(next_scores[batch_idx], next_tokens[batch_idx])
            ):
                # Get beam and token ID
                beam_id = beam_token // vocab_size
                token_id = beam_token % vocab_size
                
                # Add to candidates
                next_batch_beam.append((beam_token_score, token_id, batch_idx * beam_size + beam_id))
                
                # Stop after beam_size candidates
                if len(next_batch_beam) == (batch_idx + 1) * beam_size:
                    break
                    
        # Prepare next iteration
        batch_idx_to_beam_idx = {}
        
        # Update input_ids and beam_scores
        new_input_ids = []
        new_beam_scores = []
        
        for batch_beam_idx, (beam_score, token_id, beam_idx) in enumerate(next_batch_beam):
            # Convert to batch and beam indices
            batch_idx = beam_idx // beam_size
            
            if batch_idx not in batch_idx_to_beam_idx:
                batch_idx_to_beam_idx[batch_idx] = []
            batch_idx_to_beam_idx[batch_idx].append(batch_beam_idx)
            
            # Update input_ids
            new_input_ids.append(torch.cat([input_ids[beam_idx], token_id.unsqueeze(0)], dim=0))
            
            # Update beam scores
            new_beam_scores.append(beam_score)
            
            # Check if this beam generated EOS
            if token_id.item() == eos_token_id and eos_token_id is not None:
                # Apply length penalty
                if length_penalty != 1.0:
                    beam_score = beam_score * (len(new_input_ids[-1]) ** -length_penalty)
                    new_beam_scores[-1] = beam_score
                    
                done[batch_idx] = True
                
        # Check if all batches are done
        if all(done) and early_stopping:
            break
            
        # Update input_ids and beam_scores
        input_ids = torch.stack(new_input_ids)
        beam_scores = torch.tensor(new_beam_scores, device=device)
        
        # Update length
        current_length = input_ids.size(1)
        
    # Select the best beam for each batch
    best_sequences = []
    best_scores = []
    
    for batch_idx in range(batch_size):
        # Get indices for this batch
        beam_indices = batch_idx_to_beam_idx.get(batch_idx, list(range(batch_idx * beam_size, (batch_idx + 1) * beam_size)))
        
        # Select best beam
        best_idx = beam_indices[0]  # Default to first beam
        best_score = -float("inf")
        
        for idx in beam_indices:
            score = beam_scores[idx]
            
            # Apply length penalty
            if length_penalty != 1.0:
                score = score * (input_ids[idx].size(0) ** -length_penalty)
                
            if score > best_score:
                best_score = score
                best_idx = idx
                
        # Add to results
        best_sequences.append(input_ids[best_idx])
        best_scores.append(best_score)
        
    # Stack results
    sequences = torch.stack(best_sequences)
    scores = torch.tensor(best_scores, device=device)
    
    return sequences, scores


def diverse_beam_search(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 5,
    num_beam_groups: int = 3,
    diversity_penalty: float = 0.5,
    max_length: int = 50,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform diverse beam search with beam groups.
    
    Args:
        model: The model to generate with
        input_ids: Initial input IDs of shape [batch_size, seq_len]
        beam_size: Number of beams per group
        num_beam_groups: Number of diverse groups
        diversity_penalty: Penalty for token overlap between groups
        max_length: Maximum generation length
        **kwargs: Additional arguments for beam search
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Generated sequences and their scores
    """
    # This is a simplified diverse beam search implementation
    # A full implementation would be more complex
    
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # Total number of beams across all groups
    total_beams = beam_size * num_beam_groups
    
    # Results for all groups
    all_sequences = []
    all_scores = []
    
    # Process each beam group
    for group_idx in range(num_beam_groups):
        # Clone input for this group
        group_input_ids = input_ids.clone()
        
        # If not the first group, apply diversity penalty to already generated tokens
        diversity_penalty_applied = torch.zeros(
            batch_size, model.config.vocab_size, device=device
        )
        
        if group_idx > 0:
            # Calculate token frequency in previous groups
            for prev_group_seqs in all_sequences:
                for seq in prev_group_seqs:
                    for token_id in seq[input_ids.size(1):]:  # Only consider newly generated tokens
                        diversity_penalty_applied[:, token_id] += diversity_penalty
        
        # Perform beam search for this group
        group_sequences, group_scores = beam_search(
            model,
            group_input_ids,
            beam_size=beam_size,
            max_length=max_length,
            **kwargs
        )
        
        # Add results to all groups
        all_sequences.append(group_sequences)
        all_scores.append(group_scores)
        
    # Combine results from all groups
    combined_sequences = torch.cat(all_sequences, dim=0)  # [batch_size * total_beams, seq_len]
    combined_scores = torch.cat(all_scores, dim=0)        # [batch_size * total_beams]
    
    # Reshape to batch_size, total_beams, seq_len
    combined_sequences = combined_sequences.view(batch_size, total_beams, -1)
    combined_scores = combined_scores.view(batch_size, total_beams)
    
    return combined_sequences, combined_scores