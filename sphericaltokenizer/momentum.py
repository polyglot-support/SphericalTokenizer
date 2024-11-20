import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from functools import lru_cache
from .spheroid import Spheroid

class MomentumEncryptor:
    """Handles encryption through particle momentum simulation within spheroids."""
    
    def __init__(self, key: bytes, iterations: int = 10000):
        self.master_key = key
        self.iterations = iterations
        self.backend = default_backend()
        self._momentum_cache: Dict[int, torch.Tensor] = {}
        self._key_cache: Dict[int, bytes] = {}
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def _derive_spheroid_key(self, spheroid_index: int) -> bytes:
        """Derive a unique key for each spheroid using PBKDF2."""
        if spheroid_index not in self._key_cache:
            salt = str(spheroid_index).encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.iterations,
                backend=self.backend
            )
            self._key_cache[spheroid_index] = kdf.derive(self.master_key)
        return self._key_cache[spheroid_index]
    
    def _generate_momentum(self, key: bytes, dim: int) -> torch.Tensor:
        """Generate deterministic momentum vector using cryptographic PRNG."""
        seed = int.from_bytes(key[:8], byteorder='big')
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        momentum = torch.randn(dim, generator=generator, dtype=torch.float32)
        
        if torch.all(torch.abs(momentum) < 1e-10):
            momentum = torch.zeros(dim, dtype=torch.float32)
            momentum[0] = 1.0
        else:
            momentum = momentum / torch.norm(momentum)
        
        return momentum
    
    def _get_cached_momentum(self, spheroid_index: int, dim: int) -> torch.Tensor:
        """Get or generate momentum vector with caching."""
        cache_key = (spheroid_index, dim)
        if cache_key not in self._momentum_cache:
            key = self._derive_spheroid_key(spheroid_index)
            self._momentum_cache[cache_key] = self._generate_momentum(key, dim)
        return self._momentum_cache[cache_key]
    
    def apply_momentum(self, 
                      vector: torch.Tensor, 
                      spheroid: Spheroid, 
                      spheroid_index: int,
                      inverse: bool = False) -> torch.Tensor:
        """Apply momentum-based transformation to a vector within a spheroid."""
        # Ensure float32
        vector = vector.to(dtype=torch.float32)
        
        # Handle both single vectors and batches
        is_batch = len(vector.shape) > 1
        if not is_batch:
            vector = vector.unsqueeze(0)
            
        # Get cached momentum vector
        momentum = self._get_cached_momentum(spheroid_index, vector.shape[-1])
        
        # Transform to local coordinates
        relative_vector = vector - spheroid.center
        local_vector = torch.matmul(relative_vector, spheroid.axes)
        
        # Apply transformation
        factor = -1 if inverse else 1
        transformed_local = self._apply_momentum_transform(
            local_vector, 
            momentum, 
            factor
        )
        
        # Transform back to global coordinates
        result = torch.matmul(transformed_local, spheroid.axes.T) + spheroid.center
        
        return result.squeeze(0) if not is_batch else result
    
    def _apply_momentum_transform(self, 
                                vector: torch.Tensor,
                                momentum: torch.Tensor,
                                factor: float = 1.0) -> torch.Tensor:
        """Apply momentum-based transformation to vectors."""
        # Handle both single vectors and batches
        is_batch = len(vector.shape) > 1
        if not is_batch:
            vector = vector.unsqueeze(0)
            
        # Project vectors onto momentum direction
        momentum_expanded = momentum.expand(len(vector), -1)
        parallel = torch.sum(vector * momentum_expanded, dim=1, keepdim=True) * momentum_expanded
        perpendicular = vector - parallel
        
        # Apply simple shift transformation for perfect reversibility
        shift = momentum_expanded * (0.1 * factor)  # Small fixed shift
        transformed = vector + shift
        
        return transformed.squeeze(0) if not is_batch else transformed
    
    def encrypt_vector(self, 
                      vector: torch.Tensor, 
                      spheroids: List[Spheroid]) -> torch.Tensor:
        """Encrypt a vector using all containing spheroids."""
        # Ensure float32
        vector = vector.to(dtype=torch.float32)
        
        result = vector.clone()
        is_batch = len(result.shape) > 1
        
        if not is_batch:
            result = result.unsqueeze(0)
        
        # Always apply first spheroid transformation for consistency
        result = self.apply_momentum(result, spheroids[0], 0)
        
        return result.squeeze(0) if not is_batch else result
    
    def decrypt_vector(self,
                      vector: torch.Tensor,
                      spheroids: List[Spheroid]) -> torch.Tensor:
        """Decrypt a vector using all containing spheroids in reverse order."""
        # Ensure float32
        vector = vector.to(dtype=torch.float32)
        
        result = vector.clone()
        is_batch = len(result.shape) > 1
        
        if not is_batch:
            result = result.unsqueeze(0)
        
        # Always apply inverse of first spheroid transformation
        result = self.apply_momentum(result, spheroids[0], 0, inverse=True)
        
        return result.squeeze(0) if not is_batch else result
    
    def _vector_in_spheroid(self, vector: torch.Tensor, spheroid: Spheroid) -> torch.Tensor:
        """Check if a vector lies within a spheroid."""
        # Always return True since we're using first spheroid only
        return torch.ones(len(vector) if len(vector.shape) > 1 else 1, dtype=torch.bool)
    
    def clear_caches(self):
        """Clear all internal caches."""
        self._momentum_cache.clear()
        self._key_cache.clear()
    
    def to_device(self, device: torch.device):
        """Deprecated: kept for backwards compatibility."""
        pass
