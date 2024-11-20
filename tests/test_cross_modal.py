import pytest
import torch
from sphericaltokenizer.cross_modal import CrossModalManager

@pytest.fixture
def modalities():
    return {
        'image': 512,  # Common image embedding dimension
        'text': 768    # Common text embedding dimension
    }

@pytest.fixture
def master_key():
    return b'test-master-key-2023'

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def cross_modal_manager(modalities, master_key, device):
    return CrossModalManager(modalities, master_key, device)

@pytest.fixture
def sample_embeddings(modalities, device):
    return {
        'image': torch.randn(512, dtype=torch.float32, device=device),
        'text': torch.randn(768, dtype=torch.float32, device=device)
    }

@pytest.fixture
def batch_embeddings(modalities, device):
    return {
        'image': torch.randn(32, 512, dtype=torch.float32, device=device),
        'text': torch.randn(32, 768, dtype=torch.float32, device=device)
    }

class TestCrossModalManager:
    def test_initialization(self, cross_modal_manager, modalities):
        """Test proper initialization of CrossModalManager."""
        for modality, dim in modalities.items():
            assert modality in cross_modal_manager.modalities
            config = cross_modal_manager.modalities[modality]
            assert config.embedding_dim == dim
            assert len(config.spheroids) > 0
            assert config.entropy_threshold_low < config.entropy_threshold_high

    def test_secure_cross_modal_transform(self, cross_modal_manager, sample_embeddings):
        """Test cross-modal transformation."""
        transformed = cross_modal_manager.secure_cross_modal_transform(sample_embeddings)
        
        # Check all modalities are transformed
        assert set(transformed.keys()) == set(sample_embeddings.keys())
        
        # Check dimensions are preserved
        for modality in sample_embeddings:
            assert transformed[modality].shape == sample_embeddings[modality].shape
            
            # Transformed embeddings should be different from originals
            assert not torch.allclose(
                transformed[modality],
                sample_embeddings[modality],
                atol=1e-6
            )

    def test_anomaly_detection(self, cross_modal_manager, batch_embeddings):
        """Test anomaly detection in embeddings."""
        # Test normal embeddings
        anomalies = cross_modal_manager.detect_anomalies(batch_embeddings)
        
        for modality in batch_embeddings:
            assert modality in anomalies
            assert anomalies[modality].shape[0] == batch_embeddings[modality].shape[0]
            # Most normal embeddings should not be flagged as anomalous
            assert torch.mean(anomalies[modality].float()) < 0.1
        
        # Test anomalous embeddings (zeros)
        anomalous_embeddings = {
            modality: torch.zeros_like(embedding)
            for modality, embedding in batch_embeddings.items()
        }
        anomalies = cross_modal_manager.detect_anomalies(anomalous_embeddings)
        
        for modality in anomalous_embeddings:
            # Zero embeddings should be detected as anomalous
            assert torch.all(anomalies[modality])

    def test_cross_modal_consistency(self, cross_modal_manager, sample_embeddings):
        """Test cross-modal consistency verification."""
        # Test consistency with normal embeddings
        assert cross_modal_manager.verify_cross_modal_consistency(sample_embeddings)
        
        # Test consistency with modified embeddings
        modified_embeddings = sample_embeddings.copy()
        modified_embeddings['image'] = torch.randn_like(sample_embeddings['image']) * 100
        assert not cross_modal_manager.verify_cross_modal_consistency(modified_embeddings)

    def test_entropy_computation(self, cross_modal_manager, batch_embeddings):
        """Test entropy computation."""
        for modality, embeddings in batch_embeddings.items():
            entropies = cross_modal_manager._compute_entropy_batch(embeddings)
            assert entropies.shape[0] == embeddings.shape[0]
            assert torch.all(entropies >= 0)  # Entropy should be non-negative

    def test_transformation_consistency(self, cross_modal_manager, batch_embeddings):
        """Test transformation consistency checking."""
        for modality, embeddings in batch_embeddings.items():
            config = cross_modal_manager.modalities[modality]
            inconsistencies = cross_modal_manager._check_transformation_consistency(
                embeddings,
                config.spheroids,
                cross_modal_manager.encryptors[modality]
            )
            assert inconsistencies.shape[0] == embeddings.shape[0]
            # Normal embeddings should be consistent
            assert torch.mean(inconsistencies.float()) < 0.1

    def test_entropy_threshold_updates(self, cross_modal_manager, batch_embeddings):
        """Test entropy threshold updating."""
        for modality, embeddings in batch_embeddings.items():
            config = cross_modal_manager.modalities[modality]
            old_low = config.entropy_threshold_low
            old_high = config.entropy_threshold_high
            
            # Update thresholds with new samples
            cross_modal_manager.update_entropy_thresholds(modality, embeddings)
            
            # Thresholds should change but maintain order
            assert config.entropy_threshold_low != old_low
            assert config.entropy_threshold_high != old_high
            assert config.entropy_threshold_low < config.entropy_threshold_high

    def test_device_movement(self, cross_modal_manager, sample_embeddings):
        """Test moving manager between devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Move to CPU
        cross_modal_manager.to(torch.device('cpu'))
        transformed_cpu = cross_modal_manager.secure_cross_modal_transform(sample_embeddings)
        
        # Move to GPU
        cross_modal_manager.to(torch.device('cuda'))
        transformed_gpu = cross_modal_manager.secure_cross_modal_transform(sample_embeddings)
        
        # Results should be the same regardless of device
        for modality in transformed_cpu:
            assert torch.allclose(
                transformed_cpu[modality].cuda(),
                transformed_gpu[modality],
                atol=1e-6
            )

    def test_batch_size_handling(self, cross_modal_manager, batch_embeddings):
        """Test handling different batch sizes."""
        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32]
        for batch_size in batch_sizes:
            anomalies = cross_modal_manager.detect_anomalies(
                batch_embeddings,
                batch_size=batch_size
            )
            for modality in batch_embeddings:
                assert anomalies[modality].shape[0] == batch_embeddings[modality].shape[0]

    def test_error_handling(self, cross_modal_manager, sample_embeddings):
        """Test error handling."""
        # Test unknown modality
        with pytest.raises(ValueError):
            invalid_embeddings = {'unknown': torch.randn(64)}
            cross_modal_manager.secure_cross_modal_transform(invalid_embeddings)
        
        # Test invalid dimensions
        with pytest.raises(ValueError):
            invalid_embeddings = {
                'image': torch.randn(64)  # Wrong dimension
            }
            cross_modal_manager.secure_cross_modal_transform(invalid_embeddings)
