import torch
import numpy as np
from typing import List, Set, Optional, Dict, Any, Tuple, Union
from functools import lru_cache
from .spheroid import SpheroidGenerator
from .momentum import MomentumEncryptor
from .layers import LayerManager

class SphericalTokenizer:
    """
    Main tokenizer class that provides a secure interface for embedding transformations.
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 master_key: bytes,
                 num_spheroids: Optional[int] = None,
                 cache_size: int = 1024,
                 device: Optional[torch.device] = None):
        """
        Initialize the SphericalTokenizer.
        
        Args:
            embedding_dim: Dimension of the embedding space
            master_key: Master encryption key
            num_spheroids: Optional override for number of spheroids
            cache_size: Size of the transformation cache
            device: Optional device specification (cuda or cpu)
        """
        # Set device (GPU if available and not specified, else CPU)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedding_dim = embedding_dim
        self.generator = SpheroidGenerator(embedding_dim, num_spheroids)
        self.encryptor = MomentumEncryptor(master_key)
        self.layer_manager = LayerManager(embedding_dim)
        self.cache_size = cache_size
        
        # Move components to device
        self.generator.to_device(self.device)
        self.encryptor.to_device(self.device)
        
        # Generate base spheroids
        self.base_spheroids = self.generator.generate_spheroids()
        
        # Precompute common values for secure similarity
        self._precompute_similarity_values()
    
    def _precompute_similarity_values(self):
        """Precompute values used in secure similarity calculations."""
        self.similarity_weights = torch.linspace(0.5, 1.5, self.embedding_dim, device=self.device)
        self.similarity_phases = torch.linspace(0, 2*torch.pi, self.embedding_dim, device=self.device)
        self.cos_phases = torch.cos(self.similarity_phases)
        self.sin_phases = torch.sin(self.similarity_phases)
    
    def create_role(self, 
                   role_name: str, 
                   permissions: Set[str],
                   role_key: Optional[bytes] = None) -> None:
        """Create a new role with specified permissions."""
        key = role_key or self.encryptor.master_key
        self.layer_manager.create_layer(role_name, key, permissions)
    
    def encrypt(self, 
                vector: Union[np.ndarray, torch.Tensor],
                roles: Optional[List[str]] = None) -> Union[np.ndarray, torch.Tensor]:
        """Encrypt a vector using specified roles."""
        # Convert input to torch tensor if needed
        is_numpy = isinstance(vector, np.ndarray)
        if is_numpy:
            vector = torch.from_numpy(vector).to(self.device)
        elif not vector.device == self.device:
            vector = vector.to(self.device)
        
        if len(vector) != self.embedding_dim:
            raise ValueError(f"Vector dimension {len(vector)} does not match "
                           f"embedding dimension {self.embedding_dim}")
        
        # Apply base encryption
        result = self.encryptor.encrypt_vector(vector, self.base_spheroids)
        
        # Apply role-based layers if specified
        if roles:
            result = self.layer_manager.apply_layers(
                result, roles, self.encryptor
            )
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.cpu().numpy()
            
        return result
    
    def decrypt(self,
                vector: Union[np.ndarray, torch.Tensor],
                roles: Optional[List[str]] = None) -> Union[np.ndarray, torch.Tensor]:
        """Decrypt a vector using specified roles."""
        # Convert input to torch tensor if needed
        is_numpy = isinstance(vector, np.ndarray)
        if is_numpy:
            vector = torch.from_numpy(vector).to(self.device)
        elif not vector.device == self.device:
            vector = vector.to(self.device)
        
        if len(vector) != self.embedding_dim:
            raise ValueError(f"Vector dimension {len(vector)} does not match "
                           f"embedding dimension {self.embedding_dim}")
        
        result = vector.clone()
        
        # Remove role-based layers if specified
        if roles:
            result = self.layer_manager.apply_layers(
                result, roles, self.encryptor, inverse=True
            )
        
        # Apply base decryption
        result = self.encryptor.decrypt_vector(result, self.base_spheroids)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.cpu().numpy()
            
        return result
    
    @torch.no_grad()  # Disable gradient computation for better performance
    def transform_batch(self,
                       vectors: Union[np.ndarray, torch.Tensor],
                       roles: Optional[List[str]] = None,
                       decrypt: bool = False,
                       batch_size: int = 32) -> Union[np.ndarray, torch.Tensor]:
        """Transform a batch of vectors efficiently using GPU parallelization."""
        # Convert input to torch tensor if needed
        is_numpy = isinstance(vectors, np.ndarray)
        if is_numpy:
            vectors = torch.from_numpy(vectors).to(self.device)
        elif not vectors.device == self.device:
            vectors = vectors.to(self.device)
        
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match "
                           f"embedding dimension {self.embedding_dim}")
        
        # Process in batches for better memory efficiency
        result = torch.zeros_like(vectors, device=self.device)
        
        # Process all vectors in parallel within each batch
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            if decrypt:
                # Remove role-based layers if specified
                if roles:
                    batch = self.layer_manager.apply_layers(
                        batch, roles, self.encryptor, inverse=True
                    )
                # Apply base decryption
                batch_result = torch.stack([
                    self.encryptor.decrypt_vector(v, self.base_spheroids)
                    for v in batch
                ])
            else:
                # Apply base encryption
                batch_result = torch.stack([
                    self.encryptor.encrypt_vector(v, self.base_spheroids)
                    for v in batch
                ])
                # Apply role-based layers if specified
                if roles:
                    batch_result = self.layer_manager.apply_layers(
                        batch_result, roles, self.encryptor
                    )
            
            result[i:i + batch_size] = batch_result
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            result = result.cpu().numpy()
            
        return result
    
    def secure_similarity(self,
                         vector1: Union[np.ndarray, torch.Tensor],
                         vector2: Union[np.ndarray, torch.Tensor],
                         roles: Optional[List[str]] = None) -> float:
        """Compute similarity between vectors in secure space."""
        # Convert inputs to torch tensors if needed
        if isinstance(vector1, np.ndarray):
            vector1 = torch.from_numpy(vector1).to(self.device)
        if isinstance(vector2, np.ndarray):
            vector2 = torch.from_numpy(vector2).to(self.device)
        
        # Encrypt both vectors with additional transformations
        enc1 = self._secure_transform(self.encrypt(vector1, roles))
        enc2 = self._secure_transform(self.encrypt(vector2, roles))
        
        # Compute similarity efficiently
        return self._compute_secure_similarity(enc1, enc2).item()
    
    def _secure_transform(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply efficient secure transformation to encrypted vector."""
        # Apply vectorized non-linear transformation
        result = torch.tanh(vector)
        
        # Apply phase shift using precomputed values
        result = result * self.cos_phases + torch.roll(result, 1) * self.sin_phases
        
        # Normalize efficiently
        norm = torch.norm(result)
        if norm > 0:
            result = result / norm
            
        return result
    
    def _compute_secure_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between transformed vectors efficiently."""
        # Compute weighted dot product using precomputed weights
        weighted_dot = torch.sum(self.similarity_weights * vec1 * vec2)
        
        # Apply non-linear scaling
        return torch.tanh(weighted_dot)
    
    @torch.no_grad()  # Disable gradient computation for better performance
    def batch_secure_similarity(self,
                              vectors1: Union[np.ndarray, torch.Tensor],
                              vectors2: Union[np.ndarray, torch.Tensor],
                              roles: Optional[List[str]] = None,
                              batch_size: int = 32) -> Union[np.ndarray, torch.Tensor]:
        """Compute secure similarities for batches of vectors efficiently using GPU."""
        # Convert inputs to torch tensors if needed
        is_numpy = isinstance(vectors1, np.ndarray)
        if is_numpy:
            vectors1 = torch.from_numpy(vectors1).to(self.device)
            vectors2 = torch.from_numpy(vectors2).to(self.device)
        elif not vectors1.device == self.device:
            vectors1 = vectors1.to(self.device)
            vectors2 = vectors2.to(self.device)
        
        n_pairs = len(vectors1)
        if len(vectors2) != n_pairs:
            raise ValueError("Input batches must have the same length")
            
        similarities = torch.zeros(n_pairs, device=self.device)
        
        # Process in batches
        for i in range(0, n_pairs, batch_size):
            batch_end = min(i + batch_size, n_pairs)
            batch1 = vectors1[i:batch_end]
            batch2 = vectors2[i:batch_end]
            
            # Transform batches in parallel
            enc1 = torch.stack([self._secure_transform(self.encrypt(v, roles)) 
                              for v in batch1])
            enc2 = torch.stack([self._secure_transform(self.encrypt(v, roles)) 
                              for v in batch2])
            
            # Compute similarities for entire batch at once
            weighted_dots = torch.sum(
                self.similarity_weights.unsqueeze(0) * enc1 * enc2,
                dim=1
            )
            similarities[i:batch_end] = torch.tanh(weighted_dots)
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            similarities = similarities.cpu().numpy()
            
        return similarities
    
    def validate_access(self,
                       required_permissions: Set[str],
                       roles: List[str]) -> bool:
        """
        Validate if given roles provide required permissions.
        
        Args:
            required_permissions: Set of required permissions
            roles: List of roles to check
        
        Returns:
            Boolean indicating whether access is allowed
        """
        return self.layer_manager.validate_access(required_permissions, roles)
    
    def verify_transformation(self,
                            vector: Union[np.ndarray, torch.Tensor],
                            roles: Optional[List[str]] = None,
                            tolerance: float = 1e-10) -> bool:
        """Verify that transformation is reversible."""
        # Convert input to torch tensor if needed
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).to(self.device)
            
        encrypted = self.encrypt(vector, roles)
        decrypted = self.decrypt(encrypted, roles)
        return torch.allclose(vector, decrypted, atol=tolerance)
    
    def clear_caches(self):
        """Clear all internal caches."""
        self.generator._cache.clear()
        self.encryptor.clear_caches()
    
    def to(self, device: torch.device):
        """
        Move the tokenizer to specified device.
        
        Args:
            device: Target device (cuda or cpu)
        """
        self.device = device
        self.generator.to_device(device)
        self.encryptor.to_device(device)
        self._precompute_similarity_values()
    
    @property
    def supported_permissions(self) -> Set[str]:
        """Get set of all supported permissions across all roles."""
        return set().union(*(layer.permissions 
                           for layer in self.layer_manager.layers.values()))
    
    def get_role_permissions(self, role: str) -> Set[str]:
        """Get permissions for a specific role."""
        return (self.layer_manager.layers[role].permissions.copy() 
                if role in self.layer_manager.layers else set())
    
    def get_effective_permissions(self, roles: List[str]) -> Set[str]:
        """Get effective permissions for a combination of roles."""
        return self.layer_manager.get_effective_permissions(roles)
