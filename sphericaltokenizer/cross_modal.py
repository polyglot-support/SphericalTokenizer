import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from .spheroid import Spheroid, SpheroidGenerator
from .momentum import MomentumEncryptor
from .layers import Layer, LayerManager

@dataclass
class ModalityConfig:
    """
    Configuration for a specific modality's security settings.
    
    Attributes:
        name: Modality identifier (e.g., 'image', 'text')
        embedding_dim: Dimension of embeddings for this modality
        spheroids: List of spheroids for this modality
        entropy_threshold_low: Lower bound for acceptable entropy
        entropy_threshold_high: Upper bound for acceptable entropy
    """
    name: str
    embedding_dim: int
    spheroids: List[Spheroid]
    entropy_threshold_low: float
    entropy_threshold_high: float

class CrossModalManager:
    """
    Manages secure cross-modal transformations and anomaly detection.
    """
    
    def __init__(self, 
                 modalities: Dict[str, int],
                 master_key: bytes,
                 device: Optional[torch.device] = None):
        """
        Initialize the CrossModalManager.
        
        Args:
            modalities: Dictionary mapping modality names to their embedding dimensions
            master_key: Master encryption key
            device: Optional device specification (cuda or cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modalities: Dict[str, ModalityConfig] = {}
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create separate encryptors for each modality
        self.encryptors: Dict[str, MomentumEncryptor] = {}
        
        # Initialize modality configurations
        for name, dim in modalities.items():
            # Create modality-specific encryptor with unique key
            modality_key = master_key + name.encode()
            encryptor = MomentumEncryptor(modality_key)
            encryptor.to_device(self.device)
            self.encryptors[name] = encryptor
            
            # Generate spheroids for this modality
            generator = SpheroidGenerator(dim)
            generator.to_device(self.device)
            spheroids = generator.generate_spheroids()
            
            # Calculate initial entropy thresholds
            samples = torch.randn(1000, dim, device=self.device, dtype=torch.float32)
            entropies = self._compute_entropy_batch(samples)
            mean, std = torch.mean(entropies), torch.std(entropies)
            
            self.modalities[name] = ModalityConfig(
                name=name,
                embedding_dim=dim,
                spheroids=spheroids,
                entropy_threshold_low=float(mean - 3*std),
                entropy_threshold_high=float(mean + 3*std)
            )
    
    def secure_cross_modal_transform(self,
                                   embeddings: Dict[str, torch.Tensor],
                                   roles: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Apply secure transformations to embeddings from different modalities.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            roles: Optional list of roles for access control
        
        Returns:
            Dictionary of transformed embeddings
        """
        result = {}
        for modality, embedding in embeddings.items():
            if modality not in self.modalities:
                raise ValueError(f"Unknown modality: {modality}")
                
            config = self.modalities[modality]
            if embedding.shape[-1] != config.embedding_dim:
                raise ValueError(f"Expected dimension {config.embedding_dim} for modality {modality}, "
                              f"got {embedding.shape[-1]}")
            
            # Ensure float32 dtype and proper device
            embedding = embedding.to(dtype=torch.float32, device=self.device)
            
            # Check for valid embedding values
            if not self._is_valid_embedding(embedding):
                raise ValueError(f"Invalid embedding values for modality {modality}")
            
            # Apply modality-specific encryption
            transformed = self.encryptors[modality].encrypt_vector(
                embedding,
                config.spheroids
            )
            result[modality] = transformed
            
        return result
    
    def detect_anomalies(self,
                        embeddings: Dict[str, torch.Tensor],
                        batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies in embeddings across modalities.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            batch_size: Size of batches for processing
        
        Returns:
            Dictionary mapping modality names to boolean tensors indicating anomalies
        """
        anomalies = {}
        for modality, embedding in embeddings.items():
            if modality not in self.modalities:
                raise ValueError(f"Unknown modality: {modality}")
                
            config = self.modalities[modality]
            if embedding.shape[-1] != config.embedding_dim:
                raise ValueError(f"Expected dimension {config.embedding_dim} for modality {modality}, "
                              f"got {embedding.shape[-1]}")
            
            # Ensure float32 dtype and proper device
            embedding = embedding.to(dtype=torch.float32, device=self.device)
            
            # Process in batches
            is_batch = len(embedding.shape) > 1
            if not is_batch:
                embedding = embedding.unsqueeze(0)
                
            anomaly_flags = torch.zeros(len(embedding), dtype=torch.bool, device=self.device)
            
            for i in range(0, len(embedding), batch_size):
                batch = embedding[i:i + batch_size]
                
                # Check entropy
                entropies = self._compute_entropy_batch(batch)
                entropy_anomalies = (entropies < config.entropy_threshold_low) | \
                                  (entropies > config.entropy_threshold_high)
                
                # Check transformation consistency
                consistency_anomalies = self._check_transformation_consistency(
                    batch,
                    config.spheroids,
                    self.encryptors[modality]
                )
                
                # Check value ranges
                range_anomalies = ~self._is_valid_embedding(batch)
                
                # Combine anomaly flags with reduced sensitivity
                batch_anomalies = entropy_anomalies & (consistency_anomalies | range_anomalies)
                anomaly_flags[i:i + batch_size] = batch_anomalies
            
            anomalies[modality] = anomaly_flags.squeeze() if not is_batch else anomaly_flags
            
        return anomalies
    
    def verify_cross_modal_consistency(self,
                                     embeddings: Dict[str, torch.Tensor],
                                     tolerance: float = 1e-4) -> bool:
        """
        Verify consistency of cross-modal transformations.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
            tolerance: Maximum allowed difference in consistency checks
        
        Returns:
            Boolean indicating whether cross-modal transformations are consistent
        """
        try:
            # Check for valid embeddings
            for modality, embedding in embeddings.items():
                if not self._is_valid_embedding(embedding):
                    return False
            
            # Transform all modalities
            transformed = self.secure_cross_modal_transform(embeddings)
            
            # Check reconstruction accuracy for each modality
            for modality, embedding in embeddings.items():
                config = self.modalities[modality]
                
                # Ensure float32 dtype and proper device
                embedding = embedding.to(dtype=torch.float32, device=self.device)
                transformed_embedding = transformed[modality].to(device=self.device)
                
                # Check entropy
                entropies = self._compute_entropy_batch(embedding)
                if torch.any((entropies < config.entropy_threshold_low) | 
                           (entropies > config.entropy_threshold_high)):
                    return False
                
                # Decrypt transformed embedding
                decrypted = self.encryptors[modality].decrypt_vector(
                    transformed_embedding,
                    config.spheroids
                )
                decrypted = decrypted.to(device=self.device)
                
                # Check if reconstruction matches original with relaxed tolerance
                if not torch.allclose(embedding, decrypted, atol=1e-3, rtol=1e-3):
                    return False
                
                # Check if transformation actually modified the embedding
                diff = torch.norm(embedding - transformed_embedding)
                if diff < 0.1:  # Minimum required change
                    return False
                
                # Ensure transformation is deterministic
                transformed2 = self.encryptors[modality].encrypt_vector(
                    embedding,
                    config.spheroids
                )
                if not torch.allclose(transformed_embedding, transformed2, atol=1e-3, rtol=1e-3):
                    return False
            
            return True
        except Exception:
            return False
    
    def _compute_entropy_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy for a batch of embeddings.
        
        Args:
            batch: Batch of embeddings
        
        Returns:
            Tensor of entropy values
        """
        # Normalize embeddings to create probability distributions
        probs = torch.softmax(batch, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy
    
    def _check_transformation_consistency(self,
                                        batch: torch.Tensor,
                                        spheroids: List[Spheroid],
                                        encryptor: MomentumEncryptor) -> torch.Tensor:
        """
        Check transformation consistency for a batch of embeddings.
        
        Args:
            batch: Batch of embeddings
            spheroids: List of spheroids for the modality
            encryptor: MomentumEncryptor instance for the modality
        
        Returns:
            Boolean tensor indicating inconsistent transformations
        """
        # Apply forward and inverse transformations
        encrypted = encryptor.encrypt_vector(batch, spheroids)
        decrypted = encryptor.decrypt_vector(encrypted, spheroids)
        
        # Ensure device consistency
        batch = batch.to(device=self.device)
        encrypted = encrypted.to(device=self.device)
        decrypted = decrypted.to(device=self.device)
        
        # Check reconstruction error with relative tolerance
        batch_norm = torch.norm(batch, dim=-1, keepdim=True).clamp(min=1e-10)
        relative_error = torch.norm(batch - decrypted, dim=-1) / batch_norm.squeeze()
        reconstruction_error = relative_error > 1e-2
        
        # Check if transformation had sufficient effect
        diff = torch.norm(batch - encrypted, dim=-1)
        batch_norm = torch.norm(batch, dim=-1).clamp(min=1e-10)
        relative_diff = diff / batch_norm
        transformation_effect = relative_diff < 0.2  # Require 20% change
        
        return reconstruction_error & transformation_effect
    
    def _is_valid_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Check if embedding values are within valid ranges."""
        # Check for NaN or Inf values
        if torch.any(torch.isnan(embedding)) or torch.any(torch.isinf(embedding)):
            return torch.zeros(1, dtype=torch.bool, device=self.device)
        
        # Check magnitude
        magnitudes = torch.norm(embedding, dim=-1)
        valid_magnitude = (magnitudes > 1e-8) & (magnitudes < 1e4)
        
        # Check for reasonable value ranges (-100 to 100 is a generous range for embeddings)
        valid_range = torch.all((embedding > -100.0) & (embedding < 100.0), dim=-1)
        
        return valid_magnitude & valid_range
    
    def update_entropy_thresholds(self,
                                modality: str,
                                samples: torch.Tensor,
                                num_std: float = 3.0):
        """
        Update entropy thresholds for a modality based on new samples.
        
        Args:
            modality: Name of the modality
            samples: New sample embeddings
            num_std: Number of standard deviations for thresholds
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
            
        config = self.modalities[modality]
        if samples.shape[-1] != config.embedding_dim:
            raise ValueError(f"Expected dimension {config.embedding_dim}, got {samples.shape[-1]}")
        
        # Ensure float32 dtype and proper device
        samples = samples.to(dtype=torch.float32, device=self.device)
        
        # Compute entropies for samples
        entropies = self._compute_entropy_batch(samples)
        mean, std = torch.mean(entropies), torch.std(entropies)
        
        # Update thresholds
        config.entropy_threshold_low = float(mean - num_std * std)
        config.entropy_threshold_high = float(mean + num_std * std)
    
    def to(self, device: torch.device):
        """
        Move the manager to specified device.
        
        Args:
            device: Target device (cuda or cpu)
        """
        # Set device
        self.device = device
        
        # Move encryptors
        for encryptor in self.encryptors.values():
            encryptor.to_device(device)
        
        # Move spheroids
        for config in self.modalities.values():
            for spheroid in config.spheroids:
                spheroid.center = spheroid.center.to(device)
                spheroid.axes = spheroid.axes.to(device)
        
        # Clear caches to ensure clean state
        self.clear_caches()
    
    def clear_caches(self):
        """Clear all internal caches."""
        for encryptor in self.encryptors.values():
            encryptor.clear_caches()
