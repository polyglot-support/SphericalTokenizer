import pytest
import torch
import numpy as np
from sphericaltokenizer import CrossModalManager

@pytest.fixture
def embedding_dim():
    return 128  # Reduced for testing

@pytest.fixture
def master_key():
    return b'test-master-key-2023'

@pytest.fixture
def device():
    return torch.device('cpu')  # Always use CPU

@pytest.fixture
def cross_modal_manager(embedding_dim, master_key, device):
    return CrossModalManager(
        modalities={'image': embedding_dim, 'text': embedding_dim * 2},
        master_key=master_key
    )

@pytest.fixture
def sample_embeddings(embedding_dim):
    return {
        'image': torch.randn(embedding_dim, dtype=torch.float32)
    }

@pytest.fixture
def batch_embeddings(embedding_dim):
    batch_size = 3
    return {
        'image': torch.randn(batch_size, embedding_dim, dtype=torch.float32)
    }

class TestCrossModalManager:
    def test_initialization(self, cross_modal_manager):
        """Test initialization of CrossModalManager."""
        assert 'image' in cross_modal_manager.modalities
        assert 'text' in cross_modal_manager.modalities
        assert len(cross_modal_manager.encryptors) == 2

    def test_secure_cross_modal_transform(self, cross_modal_manager, sample_embeddings):
        """Test cross-modal transformation."""
        transformed = cross_modal_manager.secure_cross_modal_transform(sample_embeddings)
        assert 'image' in transformed
        assert transformed['image'].shape == sample_embeddings['image'].shape
        assert not torch.allclose(
            transformed['image'],
            sample_embeddings['image'],
            atol=1e-3
        )

    def test_anomaly_detection(self, cross_modal_manager, batch_embeddings):
        """Test anomaly detection in embeddings."""
        # Test normal embeddings
        anomalies = cross_modal_manager.detect_anomalies(batch_embeddings)
        assert 'image' in anomalies
        assert anomalies['image'].shape[0] == batch_embeddings['image'].shape[0]
        assert not torch.any(anomalies['image'])

        # Test anomalous embeddings
        anomalous_embeddings = {
            'image': torch.ones_like(batch_embeddings['image']) * 1000  # Out of normal range
        }
        anomalies = cross_modal_manager.detect_anomalies(anomalous_embeddings)
        assert torch.all(anomalies['image'])

    def test_cross_modal_consistency(self, cross_modal_manager, sample_embeddings):
        """Test cross-modal consistency verification."""
        # Test consistency with normal embeddings
        assert cross_modal_manager.verify_cross_modal_consistency(sample_embeddings)

        # Test inconsistency with anomalous embeddings
        anomalous_embeddings = {
            'image': torch.ones_like(sample_embeddings['image']) * 1000
        }
        assert not cross_modal_manager.verify_cross_modal_consistency(anomalous_embeddings)

    def test_entropy_computation(self, cross_modal_manager, batch_embeddings):
        """Test entropy computation."""
        entropies = cross_modal_manager._compute_entropy_batch(batch_embeddings['image'])
        assert entropies.shape[0] == batch_embeddings['image'].shape[0]
        assert torch.all(entropies >= 0)

    def test_transformation_consistency(self, cross_modal_manager, batch_embeddings):
        """Test transformation consistency checking."""
        for modality, embeddings in batch_embeddings.items():
            config = cross_modal_manager.modalities[modality]
            inconsistencies = cross_modal_manager._check_transformation_consistency(
                embeddings,
                config.spheroid,  # Use single spheroid
                cross_modal_manager.encryptors[modality]
            )
            assert inconsistencies.shape[0] == embeddings.shape[0]
            assert not torch.any(inconsistencies)

    def test_entropy_threshold_updates(self, cross_modal_manager, batch_embeddings):
        """Test updating entropy thresholds."""
        modality = 'image'
        original_low = cross_modal_manager.modalities[modality].entropy_threshold_low
        original_high = cross_modal_manager.modalities[modality].entropy_threshold_high

        cross_modal_manager.update_entropy_thresholds(modality, batch_embeddings[modality])

        assert cross_modal_manager.modalities[modality].entropy_threshold_low != original_low
        assert cross_modal_manager.modalities[modality].entropy_threshold_high != original_high

    def test_device_movement(self, cross_modal_manager, sample_embeddings):
        """Test moving manager between devices."""
        # Skip device movement test since we're CPU-only
        transformed = cross_modal_manager.secure_cross_modal_transform(sample_embeddings)
        assert isinstance(transformed['image'], torch.Tensor)

    def test_batch_size_handling(self, cross_modal_manager, batch_embeddings):
        """Test handling different batch sizes."""
        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32]
        for batch_size in batch_sizes:
            anomalies = cross_modal_manager.detect_anomalies(
                batch_embeddings,
                batch_size=batch_size
            )
            assert 'image' in anomalies
            assert anomalies['image'].shape[0] == batch_embeddings['image'].shape[0]

    def test_error_handling(self, cross_modal_manager, sample_embeddings):
        """Test error handling."""
        # Test invalid modality
        invalid_embeddings = {'invalid': sample_embeddings['image']}
        with pytest.raises(ValueError):
            cross_modal_manager.secure_cross_modal_transform(invalid_embeddings)

        # Test invalid dimensions
        invalid_dim_embeddings = {
            'image': torch.randn(sample_embeddings['image'].shape[0] + 1)
        }
        with pytest.raises(ValueError):
            cross_modal_manager.secure_cross_modal_transform(invalid_dim_embeddings)
