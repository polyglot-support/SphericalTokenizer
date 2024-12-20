import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from .spheroid import Spheroid
from .momentum import MomentumEncryptor

@dataclass
class Layer:
    """
    Represents a security layer with its own spheroid configuration.
    
    Attributes:
        name: Layer identifier
        spheroid: Single spheroid for this layer
        permissions: Set of allowed operations
        key: Layer-specific encryption key
    """
    name: str
    spheroid: Spheroid
    permissions: Set[str]
    key: bytes

class LayerManager:
    """
    Manages RBAC layers and their compositions for secure embedding transformations.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize the LayerManager.
        
        Args:
            embedding_dim: Dimension of the embedding space
        """
        self.embedding_dim = embedding_dim
        self.layers: Dict[str, Layer] = {}
        self.layer_order: List[str] = []
    
    def create_layer(self, 
                    name: str, 
                    key: bytes,
                    permissions: Optional[Set[str]] = None) -> Layer:
        """
        Create a new security layer.
        
        Args:
            name: Layer identifier
            key: Encryption key for the layer
            permissions: Set of allowed operations
        
        Returns:
            Created Layer object
        """
        if name in self.layers:
            raise ValueError(f"Layer '{name}' already exists")
            
        # Generate spheroid for this layer
        from .spheroid import SpheroidGenerator  # Import here to avoid circular import
        generator = SpheroidGenerator(self.embedding_dim)
        spheroid = generator.generate_spheroids()[0]  # Use only first spheroid
        
        # Create layer with specified permissions
        layer = Layer(
            name=name,
            spheroid=spheroid,
            permissions=permissions or set(),
            key=key
        )
        
        self.layers[name] = layer
        self.layer_order.append(name)
        return layer
    
    def add_permission(self, layer_name: str, permission: str) -> None:
        """
        Add a permission to a layer.
        
        Args:
            layer_name: Name of the layer
            permission: Permission to add
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' does not exist")
        
        self.layers[layer_name].permissions.add(permission)
    
    def remove_permission(self, layer_name: str, permission: str) -> None:
        """
        Remove a permission from a layer.
        
        Args:
            layer_name: Name of the layer
            permission: Permission to remove
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' does not exist")
        
        self.layers[layer_name].permissions.discard(permission)
    
    def has_permission(self, layer_name: str, permission: str) -> bool:
        """
        Check if a layer has a specific permission.
        
        Args:
            layer_name: Name of the layer
            permission: Permission to check
        
        Returns:
            Boolean indicating whether the layer has the permission
        """
        if layer_name not in self.layers:
            return False
        
        return permission in self.layers[layer_name].permissions
    
    @torch.no_grad()
    def apply_layer(self,
                   vector: torch.Tensor,
                   layer_name: str,
                   encryptor: MomentumEncryptor,
                   inverse: bool = False) -> torch.Tensor:
        """
        Apply a layer's transformation to a vector or batch of vectors.
        
        Args:
            vector: Input vector or batch of vectors
            layer_name: Name of the layer to apply
            encryptor: MomentumEncryptor instance
            inverse: Whether to apply inverse transformation
        
        Returns:
            Transformed vector(s)
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' does not exist")
        
        layer = self.layers[layer_name]
        
        # Apply transformation using single spheroid
        result = encryptor.apply_momentum(
            vector,
            layer.spheroid,
            self.layer_order.index(layer_name),
            inverse=inverse
        )
        
        return result
    
    @torch.no_grad()
    def apply_layers(self,
                    vector: torch.Tensor,
                    layer_names: List[str],
                    encryptor: MomentumEncryptor,
                    inverse: bool = False) -> torch.Tensor:
        """
        Apply multiple layers' transformations to a vector or batch of vectors.
        
        Args:
            vector: Input vector or batch of vectors
            layer_names: Names of layers to apply
            encryptor: MomentumEncryptor instance
            inverse: Whether to apply inverse transformations
        
        Returns:
            Transformed vector(s)
        """
        result = vector.clone()
        
        # Apply layers in reverse order for inverse transformation
        if inverse:
            layer_names = reversed(layer_names)
        
        for name in layer_names:
            result = self.apply_layer(result, name, encryptor, inverse)
        
        return result
    
    def verify_layer_composition(self, 
                               vector: torch.Tensor,
                               layer_names: List[str],
                               encryptor: MomentumEncryptor,
                               tolerance: float = 1e-6) -> bool:  # Relaxed tolerance
        """
        Verify that layer composition is reversible.
        
        Args:
            vector: Test vector
            layer_names: Names of layers to verify
            encryptor: MomentumEncryptor instance
            tolerance: Maximum allowed difference
        
        Returns:
            Boolean indicating whether composition is valid
        """
        # Apply forward transformation
        encrypted = self.apply_layers(vector, layer_names, encryptor)
        
        # Apply inverse transformation
        decrypted = self.apply_layers(encrypted, layer_names, encryptor, inverse=True)
        
        # Check if original vector is recovered within tolerance
        return torch.allclose(vector, decrypted, atol=tolerance)
    
    def get_effective_permissions(self, layer_names: List[str]) -> Set[str]:
        """
        Get the intersection of permissions from multiple layers.
        
        Args:
            layer_names: Names of layers to check
        
        Returns:
            Set of permissions common to all specified layers
        """
        if not layer_names:
            return set()
        
        # Start with all permissions from first layer
        result = self.layers[layer_names[0]].permissions.copy()
        
        # Intersect with permissions from other layers
        for name in layer_names[1:]:
            if name in self.layers:
                result.intersection_update(self.layers[name].permissions)
        
        return result
    
    def validate_access(self,
                       required_permissions: Set[str],
                       layer_names: List[str]) -> bool:
        """
        Validate if given layers provide required permissions.
        
        Args:
            required_permissions: Set of required permissions
            layer_names: Names of layers to check
        
        Returns:
            Boolean indicating whether access is allowed
        """
        effective_permissions = self.get_effective_permissions(layer_names)
        return required_permissions.issubset(effective_permissions)
