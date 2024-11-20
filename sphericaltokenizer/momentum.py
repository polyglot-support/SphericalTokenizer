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
    """
    Handles encryption through particle momentum simulation within spheroids.
    """
    
    def __init__(self, key: bytes, iterations: int = 10000):
        """
        Initialize the MomentumEncryptor.
        
        Args:
            key: Master encryption key
            iterations: Number of iterations for key derivation
        """
        self.master_key = key
        self.iterations = iterations
        self.backend = default_backend()
        self._momentum_cache: Dict[int, torch.Tensor] = {}
        self._key_cache: Dict[int, bytes] = {}
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @lru_cache(maxsize=128)
    def _derive_spheroid_key(self, spheroid_index: int) -> bytes:
        """
        Derive a unique key for each spheroid using PBKDF2 with caching.
        
        Args:
            spheroid_index: Index of the spheroid
        
        Returns:
            Derived key bytes
        """
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
        """
        Generate deterministic momentum vector using cryptographic PRNG.
        
        Args:
            key: Spheroid-specific key
            dim: Dimension of the momentum vector
        
        Returns:
            Momentum vector as torch tensor
        """
        # Use key bytes to seed torch's random generator
        seed = int.from_bytes(key[:8], byteorder='big')
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        
        # Generate random vector from normal distribution
        momentum = torch.randn(dim, device=self.device, generator=generator)
        
        # Ensure non-zero vector
        if torch.all(torch.abs(momentum) < 1e-10):
            momentum = torch.zeros(dim, device=self.device)
            momentum[0] = 1.0
        else:
            momentum = momentum / torch.norm(momentum)
        
        return momentum
    
    def _get_cached_momentum(self, spheroid_index: int, dim: int) -> torch.Tensor:
        """
        Get or generate momentum vector with caching.
        
        Args:
            spheroid_index: Index of the spheroid
            dim: Dimension of the vector
        
        Returns:
            Momentum vector
        """
        if spheroid_index not in self._momentum_cache:
            key = self._derive_spheroid_key(spheroid_index)
            self._momentum_cache[spheroid_index] = self._generate_momentum(key, dim)
        return self._momentum_cache[spheroid_index]
    
    def apply_momentum(self, 
                      vector: torch.Tensor, 
                      spheroid: Spheroid, 
                      spheroid_index: int,
                      inverse: bool = False) -> torch.Tensor:
        """
        Apply momentum-based transformation to a vector within a spheroid.
        
        Args:
            vector: Input vector to transform
            spheroid: Spheroid containing the vector
            spheroid_index: Index of the spheroid
            inverse: Whether to apply inverse transformation
        
        Returns:
            Transformed vector
        """
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
            spheroid.radius,
            factor
        )
        
        # Transform back to global coordinates
        result = torch.matmul(transformed_local, spheroid.axes.T) + spheroid.center
        
        return result.squeeze(0) if not is_batch else result
    
    def _apply_momentum_transform(self, 
                                vector: torch.Tensor, 
                                momentum: torch.Tensor,
                                radius: float,
                                factor: float = 1.0) -> torch.Tensor:
        """
        Apply the actual momentum-based transformation.
        
        Args:
            vector: Vector in local coordinates
            momentum: Momentum vector
            radius: Spheroid radius
            factor: Direction factor (1 for forward, -1 for inverse)
        
        Returns:
            Transformed vector
        """
        # Handle both single vectors and batches
        is_batch = len(vector.shape) > 1
        if not is_batch:
            vector = vector.unsqueeze(0)
            
        # Calculate relative position and angle efficiently
        relative_pos = torch.norm(vector, dim=1) / radius
        angle = torch.pi * relative_pos * factor
        
        # Compute rotation matrix efficiently
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Compute outer product once
        momentum_outer = torch.outer(momentum, momentum)
        
        # Build rotation matrix efficiently
        eye = torch.eye(len(momentum), device=self.device)
        cross_product = torch.cross(
            momentum.unsqueeze(0),
            eye
        ).squeeze(0)
        
        # Apply rotation with broadcasting for batch support
        rotation = (cos_angle.unsqueeze(-1).unsqueeze(-1) * eye +
                   sin_angle.unsqueeze(-1).unsqueeze(-1) * cross_product +
                   (1 - cos_angle).unsqueeze(-1).unsqueeze(-1) * momentum_outer)
        
        # Add non-linear component with broadcasting
        non_linear = torch.sin(relative_pos.unsqueeze(-1) * torch.pi) * momentum * 0.2
        
        # Apply transformation efficiently
        result = torch.matmul(vector, rotation.transpose(-2, -1)) + non_linear * radius
        
        return result.squeeze(0) if not is_batch else result
    
    def encrypt_vector(self, 
                      vector: torch.Tensor, 
                      spheroids: List[Spheroid]) -> torch.Tensor:
        """
        Encrypt a vector using all containing spheroids.
        
        Args:
            vector: Vector to encrypt
            spheroids: List of all spheroids
        
        Returns:
            Encrypted vector
        """
        result = vector.clone()
        
        # Process spheroids
        for i, spheroid in enumerate(spheroids):
            if self._vector_in_spheroid(result, spheroid):
                result = self.apply_momentum(result, spheroid, i)
        
        return result
    
    def decrypt_vector(self,
                      vector: torch.Tensor,
                      spheroids: List[Spheroid]) -> torch.Tensor:
        """
        Decrypt a vector using all containing spheroids in reverse order.
        
        Args:
            vector: Vector to decrypt
            spheroids: List of all spheroids
        
        Returns:
            Decrypted vector
        """
        result = vector.clone()
        
        # Process spheroids in reverse order
        for i, spheroid in reversed(list(enumerate(spheroids))):
            if self._vector_in_spheroid(result, spheroid):
                result = self.apply_momentum(result, spheroid, i, inverse=True)
        
        return result
    
    def _vector_in_spheroid(self, vector: torch.Tensor, spheroid: Spheroid) -> bool:
        """
        Check if a vector lies within a spheroid.
        
        Args:
            vector: Vector to check
            spheroid: Spheroid to test against
        
        Returns:
            Boolean indicating whether vector is in spheroid
        """
        # Handle both single vectors and batches
        is_batch = len(vector.shape) > 1
        if not is_batch:
            vector = vector.unsqueeze(0)
            
        # Efficient containment check
        relative_vector = vector - spheroid.center
        local_vector = torch.matmul(relative_vector, spheroid.axes)
        scaled_vector = local_vector / (spheroid.radius * spheroid.eccentricity)
        result = torch.sum(scaled_vector ** 2, dim=1) <= 1.0
        
        return result.squeeze(0) if not is_batch else result
    
    def clear_caches(self):
        """Clear all internal caches."""
        self._momentum_cache.clear()
        self._key_cache.clear()
        self._derive_spheroid_key.cache_clear()
    
    def to_device(self, device: torch.device):
        """
        Move the encryptor to specified device.
        
        Args:
            device: Target device (cuda or cpu)
        """
        self.device = device
        self._momentum_cache.clear()
        
    def numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on current device."""
        return torch.from_numpy(array).to(self.device)
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
