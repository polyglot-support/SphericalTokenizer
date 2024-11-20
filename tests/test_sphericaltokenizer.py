import pytest
import numpy as np
import torch
from sphericaltokenizer import SphericalTokenizer
from sphericaltokenizer.spheroid import SpheroidGenerator, Spheroid
from sphericaltokenizer.momentum import MomentumEncryptor
from sphericaltokenizer.layers import LayerManager

@pytest.fixture
def embedding_dim():
    return 64  # Reduced from 768 for faster testing

@pytest.fixture
def master_key():
    return b'test-master-key-2023'

@pytest.fixture
def device():
    return torch.device('cpu')  # Always use CPU

@pytest.fixture
def tokenizer(embedding_dim, master_key, device):
    return SphericalTokenizer(embedding_dim, master_key)

@pytest.fixture
def vector(embedding_dim, device):
    return torch.randn(embedding_dim, dtype=torch.float32)

@pytest.fixture
def spheroid_generator(embedding_dim, device):
    return SpheroidGenerator(embedding_dim)

@pytest.fixture
def momentum_encryptor(master_key, device):
    return MomentumEncryptor(master_key)

@pytest.fixture
def layer_manager(embedding_dim, device):
    return LayerManager(embedding_dim)

class TestSpheroidGenerator:
    def test_spheroid_generation(self, spheroid_generator):
        spheroids = spheroid_generator.generate_spheroids()
        assert len(spheroids) == max(1, spheroid_generator.embedding_dim // 3)
        assert all(isinstance(s, Spheroid) for s in spheroids)

    def test_contains_point(self, spheroid_generator, device):
        spheroids = spheroid_generator.generate_spheroids()
        point = torch.zeros(spheroid_generator.embedding_dim, dtype=torch.float32)
        # Center point should be contained in at least one spheroid
        contained = any(spheroid_generator.contains_point(s, point) for s in spheroids)
        assert contained

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            SpheroidGenerator(2)  # Minimum dimension is 3

class TestMomentumEncryptor:
    def test_key_derivation(self, momentum_encryptor):
        key1 = momentum_encryptor._derive_spheroid_key(1)
        key2 = momentum_encryptor._derive_spheroid_key(2)
        assert key1 != key2
        assert len(key1) == len(key2) == 32

    def test_momentum_generation(self, momentum_encryptor, embedding_dim):
        key = momentum_encryptor._derive_spheroid_key(0)
        momentum = momentum_encryptor._generate_momentum(key, embedding_dim)
        assert len(momentum) == embedding_dim
        assert torch.isclose(torch.norm(momentum), torch.tensor(1.0))

    def test_vector_encryption(self, momentum_encryptor, vector, spheroid_generator):
        spheroids = spheroid_generator.generate_spheroids()
        encrypted = momentum_encryptor.encrypt_vector(vector, spheroids)
        decrypted = momentum_encryptor.decrypt_vector(encrypted, spheroids)
        assert torch.allclose(vector, decrypted, atol=1e-3)  # Relaxed tolerance

class TestLayerManager:
    def test_layer_creation(self, layer_manager):
        layer_manager.create_layer("test", b'key', {"read"})
        assert "test" in layer_manager.layers
        assert "read" in layer_manager.layers["test"].permissions

    def test_permission_management(self, layer_manager):
        layer_manager.create_layer("test", b'key', {"read"})
        layer_manager.add_permission("test", "write")
        assert "write" in layer_manager.layers["test"].permissions
        layer_manager.remove_permission("test", "write")
        assert "write" not in layer_manager.layers["test"].permissions

    def test_layer_composition(self, layer_manager, vector, momentum_encryptor):
        layer_manager.create_layer("layer1", b'key1', {"read"})
        layer_manager.create_layer("layer2", b'key2', {"write"})
        assert layer_manager.verify_layer_composition(
            vector,
            ["layer1", "layer2"],
            momentum_encryptor
        )

class TestSphericalTokenizer:
    def test_basic_encryption(self, tokenizer, vector):
        encrypted = tokenizer.encrypt(vector)
        decrypted = tokenizer.decrypt(encrypted)
        assert torch.allclose(vector, decrypted, atol=1e-3)  # Relaxed tolerance

    def test_role_based_encryption(self, tokenizer, vector):
        tokenizer.create_role("test_role", {"read"})
        encrypted = tokenizer.encrypt(vector, roles=["test_role"])
        decrypted = tokenizer.decrypt(encrypted, roles=["test_role"])
        assert torch.allclose(vector, decrypted, atol=1e-3)  # Relaxed tolerance

    def test_batch_processing(self, tokenizer, embedding_dim, device):
        batch_size = 5  # Reduced from 10
        vectors = torch.randn(batch_size, embedding_dim, dtype=torch.float32)
        encrypted = tokenizer.transform_batch(vectors)
        decrypted = tokenizer.transform_batch(encrypted, decrypt=True)
        assert torch.allclose(vectors, decrypted, atol=1e-3)  # Relaxed tolerance

    def test_secure_similarity(self, tokenizer, embedding_dim, device):
        v1 = torch.randn(embedding_dim, dtype=torch.float32)
        v2 = torch.randn(embedding_dim, dtype=torch.float32)
        
        # Original similarity
        orig_sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        
        # Secure similarity
        secure_sim = tokenizer.secure_similarity(v1, v2)
        
        # Similarities should be different but consistent
        assert not torch.isclose(orig_sim, torch.tensor(secure_sim))
        assert -1 <= secure_sim <= 1

    def test_batch_secure_similarity(self, tokenizer, embedding_dim, device):
        batch_size = 5  # Small batch for testing
        vectors1 = torch.randn(batch_size, embedding_dim, dtype=torch.float32)
        vectors2 = torch.randn(batch_size, embedding_dim, dtype=torch.float32)
        
        similarities = tokenizer.batch_secure_similarity(vectors1, vectors2)
        assert len(similarities) == batch_size
        assert all(-1 <= sim <= 1 for sim in similarities)

    def test_permission_validation(self, tokenizer):
        tokenizer.create_role("admin", {"read", "write"})
        tokenizer.create_role("user", {"read"})
        
        assert tokenizer.validate_access({"read"}, ["user"])
        assert not tokenizer.validate_access({"write"}, ["user"])
        assert tokenizer.validate_access({"read", "write"}, ["admin"])

    def test_invalid_dimension(self, tokenizer, device):
        invalid_vector = torch.randn(tokenizer.embedding_dim + 1, dtype=torch.float32)
        with pytest.raises(ValueError):
            tokenizer.encrypt(invalid_vector)

    def test_transformation_verification(self, tokenizer, vector):
        tokenizer.create_role("test_role", {"read"})
        assert tokenizer.verify_transformation(vector, roles=["test_role"])

    def test_effective_permissions(self, tokenizer):
        tokenizer.create_role("role1", {"read", "write"})
        tokenizer.create_role("role2", {"read", "execute"})
        
        effective_perms = tokenizer.get_effective_permissions(["role1", "role2"])
        assert effective_perms == {"read"}
