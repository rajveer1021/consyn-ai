"""
Test suite for Consyn AI training components
"""

import pytest
import torch

from consyn.model.config import ConsynVerseConfig
from consyn.model.architecture import ConsynLMHeadModel
from consyn.training.dataset import get_dataset
from consyn.training.optimizer import get_optimizer
from consyn.training.scheduler import get_scheduler


class TestDatasetHandling:
    def test_dataset_creation(self):
        """Test dataset creation with dummy data"""
        config = ConsynVerseConfig()
        
        # Create dummy tokenizer (you might want to import the actual tokenizer)
        class DummyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3, 4, 5]  # Dummy token ids
            
            def __call__(self, text):
                return self.encode(text)
        
        tokenizer = DummyTokenizer()
        
        # Test text dataset creation
        dataset = get_dataset(
            data_path="tests/dummy_data.txt",  # You'd need to create this
            tokenizer=tokenizer,
            dataset_type="text",
            block_size=config.max_position_embeddings
        )
        
        assert dataset is not None


class TestOptimization:
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        
        optimizer = get_optimizer(
            model=model,
            lr=5e-5,
            weight_decay=0.01
        )
        
        assert optimizer is not None
        assert len(optimizer.param_groups) > 0


class TestScheduler:
    def test_scheduler_creation(self):
        """Test learning rate scheduler creation"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        optimizer = get_optimizer(model=model)
        
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        assert scheduler is not None


class TestTrainingWorkflow:
    def test_basic_training_step(self):
        """Simulate a basic training step"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        optimizer = get_optimizer(model=model)
        
        # Create dummy input
        input_ids = torch.randint(0, config.vocab_size, (4, 32))
        
        # Forward pass
        outputs = model(input_ids)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0