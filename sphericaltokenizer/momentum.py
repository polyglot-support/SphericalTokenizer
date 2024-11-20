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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        generator = torch.Generator(device='cpu')  # Generate on CPU then move to device
        generator.manual_seed(seed)
        
        momentum = torch.randn(dim, generator=generator, dtype=torch.float32)
        momentum = momentum.to(self.device)
        
        if torch.all(torch.abs(momentum) < 1e-10):
            momentum = torch.zeros(dim, device=self.device, dtype=torch.float32)
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
        # Move input to current device and ensure float32
        vector = vector.to(dtype=torch.float32, device=self.device)
        
        # Handle both single vectors and batches
        is_batch = len(vector.shape) > 1
        if not is_batch:
            vector = vector.unsqueeze(0)
            
        # Get cached momentum vector
        momentum = self._get_cached_momentum(spheroid_index, vector.shape[-1])
        
        # Move spheroid to current device if needed
        spheroid_center = spheroid.center.to(dtype=torch.float32, device=self.device)
        spheroid_axes = spheroid.axes.to(dtype=torch.float32, device=self.device)
        
        # Transform to local coordinates
        relative_vector = vector - spheroid_center
        local_vector = torch.matmul(relative_vector, spheroid_axes)
        
        # Apply transformation
        factor = -1 if inverse else 1
        transformed_local = self._apply_momentum_transform(
            local_vector, 
            momentum, 
            spheroid.radius,
            factor
        )
        
        # Transform back to global coordinates
        result = torch.matmul(transformed_local, spheroid_axes.T) + spheroid_center
        
        return result.squeeze(0) if not is_batch else result
    
    def _apply_momentum_transform(self, 
                                vector: torch.Tensor, 
                                momentum: torch.Tensor,
                                radius: float,
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
        
        # Apply scaling transformation
        scale = torch.tensor(2.0, device=self.device)  # Double the magnitude
        if factor > 0:
            # Scale parallel component
            transformed = perpendicular + scale * parallel
        else:
            # Inverse scaling
            transformed = perpendicular + parallel / scale
        
        # Ensure minimum transformation magnitude
        if factor > 0:
            diff = transformed - vector
            diff_norm = torch.norm(diff, dim=1, keepdim=True)
            min_change = radius * 0.2  # 20% of radius as minimum change
            scale_factor = torch.maximum(
                torch.ones_like(diff_norm, device=self.device),
                min_change / diff_norm.clamp(min=1e-10)
            )
            transformed = vector + diff * scale_factor
        
        return transformed.squeeze(0) if not is_batch else transformed
    
    def encrypt_vector(self, 
                      vector: torch.Tensor, 
                      spheroids: List[Spheroid]) -> torch.Tensor:
        """Encrypt a vector using all containing spheroids."""
        # Move input to current device and ensure float32
        vector = vector.to(dtype=torch.float32, device=self.device)
        
        result = vector.clone()
        is_batch = len(result.shape) > 1
        
        if not is_batch:
            result = result.unsqueeze(0)
        
        # Always apply at least one transformation
        transformed = False
        
        # Process spheroids
        for i, spheroid in enumerate(spheroids):
            # Check which vectors in batch are in spheroid
            in_spheroid = self._vector_in_spheroid(result, spheroid)
            if torch.any(in_spheroid):
                # Apply momentum only to vectors that are in spheroid
                transformed_vectors = self.apply_momentum(
                    result[in_spheroid],
                    spheroid,
                    i
                )
                result[in_spheroid] = transformed_vectors
                transformed = True
        
        # If no spheroid contained the vector, use the first one
        if not transformed:
            result = self.apply_momentum(result, spheroids[0], 0)
        
        return result.squeeze(0) if not is_batch else result
    
    def decrypt_vector(self,
                      vector: torch.Tensor,
                      spheroids: List[Spheroid]) -> torch.Tensor:
        """Decrypt a vector using all containing spheroids in reverse order."""
        # Move input to current device and ensure float32
        vector = vector.to(dtype=torch.float32, device=self.device)
        
        result = vector.clone()
        is_batch = len(result.shape) > 1
        
        if not is_batch:
            result = result.unsqueeze(0)
        
        # Track if any transformation was applied
        transformed = False
        
        # Process spheroids in reverse order
        for i, spheroid in reversed(list(enumerate(spheroids))):
            # Check which vectors in batch are in spheroid
            in_spheroid = self._vector_in_spheroid(result, spheroid)
            if torch.any(in_spheroid):
                # Apply inverse momentum only to vectors that are in spheroid
                transformed_vectors = self.apply_momentum(
                    result[in_spheroid],
                    spheroid,
                    i,
                    inverse=True
                )
                result[in_spheroid] = transformed_vectors
                transformed = True
        
        # If no spheroid contained the vector, use the first one
        if not transformed:
            result = self.apply_momentum(result, spheroids[0], 0, inverse=True)
        
        return result.squeeze(0) if not is_batch else result
    
    def _vector_in_spheroid(self, vector: torch.Tensor, spheroid: Spheroid) -> torch.Tensor:
        """Check if a vector lies within a spheroid."""
        # Move spheroid to current device if needed
        spheroid_center = spheroid.center.to(dtype=torch.float32, device=self.device)
        spheroid_axes = spheroid.axes.to(dtype=torch.float32, device=self.device)
        
        # Handle both single vectors and batches
        is_batch = len(vector.shape) > 1
        if not is_batch:
            vector = vector.unsqueeze(0)
            
        # Efficient containment check
        relative_vector = vector - spheroid_center
        local_vector = torch.matmul(relative_vector, spheroid_axes)
        scaled_vector = local_vector / (spheroid.radius * spheroid.eccentricity)
        result = torch.sum(scaled_vector ** 2, dim=1) <= 2.0  # Increased containment threshold
        
        return result.squeeze(0) if not is_batch else result
    
    def clear_caches(self):
        """Clear all internal caches."""
        self._momentum_cache.clear()
        self._key_cache.clear()
    
    def to_device(self, device: torch.device):
        """Move the encryptor to specified device."""
        self.device = device
        self._momentum_cache = {
            k: v.to(device) for k, v in self._momentum_cache.items()
        }
    
    def numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on current device."""
        return torch.from_numpy(array).to(dtype=torch.float32, device=self.device)
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
