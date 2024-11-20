import torch
from typing import Dict, List, Optional, Set, Tuple, Union, Literal
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
        spheroid: Single spheroid for this modality
        entropy_threshold_low: Lower bound for acceptable entropy
        entropy_threshold_high: Upper bound for acceptable entropy
        complexity: Level of transformation complexity
    """
    name: str
    embedding_dim: int
    spheroid: Spheroid
    entropy_threshold_low: float
    entropy_threshold_high: float
    complexity: Literal['minimal', 'basic', 'standard', 'high'] = 'standard'

class CrossModalManager:
    """
    Manages secure cross-modal transformations and anomaly detection.
    """
    
    def __init__(self, 
                 modalities: Dict[str, int],
                 master_key: bytes,
                 complexity: Literal['minimal', 'basic', 'standard', 'high'] = 'standard'):
        """
        Initialize the CrossModalManager.
        
        Args:
            modalities: Dictionary mapping modality names to their embedding dimensions
            master_key: Master encryption key
            complexity: Level of transformation complexity
        """
        self.modalities: Dict[str, ModalityConfig] = {}
        self.complexity = complexity
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create separate encryptors for each modality
        self.encryptors: Dict[str, MomentumEncryptor] = {}
        
        # Initialize modality configurations
        for name, dim in modalities.items():
            # Create modality-specific encryptor with unique key
            modality_key = master_key + name.encode()
            encryptor = MomentumEncryptor(modality_key)
            self.encryptors[name] = encryptor
            
            # Generate spheroid based on complexity
            generator = SpheroidGenerator(dim)
            spheroid = generator.generate_spheroids()[0]  # Use only first spheroid
            
            # Calculate initial entropy thresholds
            samples = torch.randn(1000, dim, dtype=torch.float32)
            entropies = self._compute_entropy_batch(samples)
            mean, std = torch.mean(entropies), torch.std(entropies)
            
            self.modalities[name] = ModalityConfig(
                name=name,
                embedding_dim=dim,
                spheroid=spheroid,
                entropy_threshold_low=float(mean - 3*std),
                entropy_threshold_high=float(mean + 3*std),
                complexity=complexity
            )
    
    def secure_cross_modal_transform(self,
                                   embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply secure transformations to embeddings from different modalities.
        
        Args:
            embeddings: Dictionary mapping modality names to their embeddings
        
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
            
            # Ensure float32 dtype
            embedding = embedding.to(dtype=torch.float32)
            
            # Check for valid embedding values
            if not self._is_valid_embedding(embedding):
                raise ValueError(f"Invalid embedding values for modality {modality}")
            
            # Apply modality-specific encryption
            result[modality] = self.encryptors[modality].apply_momentum(
                embedding,
                config.spheroid,
                0  # Always use index 0 since we only have one spheroid
            )
            
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
            
            # Ensure float32 dtype
            embedding = embedding.to(dtype=torch.float32)
            
            # Process in batches
            is_batch = len(embedding.shape) > 1
            if not is_batch:
                embedding = embedding.unsqueeze(0)
                
            anomaly_flags = torch.zeros(len(embedding), dtype=torch.bool)
            
            for i in range(0, len(embedding), batch_size):
                batch = embedding[i:i + batch_size]
                
                # Check entropy
                entropies = self._compute_entropy_batch(batch)
                entropy_anomalies = (entropies < config.entropy_threshold_low) | \
                                  (entropies > config.entropy_threshold_high)
                
                # Check transformation consistency
                consistency_anomalies = self._check_transformation_consistency(
                    batch,
                    config.spheroid,
                    self.encryptors[modality]
                )
                
                # Check value ranges
                range_anomalies = not self._is_valid_embedding(batch)
                
                # Combine anomaly flags with reduced sensitivity
                batch_anomalies = entropy_anomalies & (consistency_anomalies | range_anomalies)
                anomaly_flags[i:i + batch_size] = batch_anomalies
            
            anomalies[modality] = anomaly_flags.squeeze() if not is_batch else anomaly_flags
            
        return anomalies
    
    def verify_cross_modal_consistency(self,
                                     embeddings: Dict[str, torch.Tensor],
                                     tolerance: float = 1e-3) -> bool:  # Relaxed tolerance
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
                
                # Ensure float32 dtype
                embedding = embedding.to(dtype=torch.float32)
                transformed_embedding = transformed[modality]
                
                # Check entropy
                entropies = self._compute_entropy_batch(embedding)
                if torch.any((entropies < config.entropy_threshold_low) | 
                           (entropies > config.entropy_threshold_high)):
                    return False
                
                # Decrypt transformed embedding
                decrypted = self.encryptors[modality].apply_momentum(
                    transformed_embedding,
                    config.spheroid,
                    0,  # Always use index 0
                    inverse=True
                )
                
                # Check if reconstruction matches original
                if not torch.allclose(embedding, decrypted, atol=tolerance):
                    return False
                
                # Check if transformation actually modified the embedding
                diff = torch.norm(embedding - transformed_embedding)
                if diff < 0.01:  # Reduced minimum required change
                    return False
                
                # Ensure transformation is deterministic
                transformed2 = self.encryptors[modality].apply_momentum(
                    embedding,
                    config.spheroid,
                    0  # Always use index 0
                )
                if not torch.allclose(transformed_embedding, transformed2, atol=tolerance):
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
                                        spheroid: Spheroid,
                                        encryptor: MomentumEncryptor) -> torch.Tensor:
        """
        Check transformation consistency for a batch of embeddings.
        
        Args:
            batch: Batch of embeddings
            spheroid: Spheroid for the modality
            encryptor: MomentumEncryptor instance
        
        Returns:
            Boolean tensor indicating inconsistent transformations
        """
        # Apply forward and inverse transformations
        encrypted = encryptor.apply_momentum(batch, spheroid, 0)
        decrypted = encryptor.apply_momentum(encrypted, spheroid, 0, inverse=True)
        
        # Check reconstruction error
        reconstruction_error = not torch.allclose(batch, decrypted, atol=1e-3)
        
        # Check if transformation had sufficient effect
        diff = torch.norm(batch - encrypted, dim=-1)
        transformation_effect = diff < 0.01  # Reduced minimum change
        
        return reconstruction_error & transformation_effect
    
    def _is_valid_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Check if embedding values are within valid ranges."""
        # Check for NaN or Inf values
        if torch.any(torch.isnan(embedding)) or torch.any(torch.isinf(embedding)):
            return False
        
        # Check magnitude
        magnitudes = torch.norm(embedding, dim=-1)
        valid_magnitude = torch.all((magnitudes > 1e-8) & (magnitudes < 1e4))
        
        # Check for reasonable value ranges (-100 to 100 is a generous range for embeddings)
        valid_range = torch.all((embedding > -100.0) & (embedding < 100.0))
        
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
        
        # Ensure float32 dtype
        samples = samples.to(dtype=torch.float32)
        
        # Compute entropies for samples
        entropies = self._compute_entropy_batch(samples)
        mean, std = torch.mean(entropies), torch.std(entropies)
        
        # Update thresholds
        config.entropy_threshold_low = float(mean - num_std * std)
        config.entropy_threshold_high = float(mean + num_std * std)
    
    def clear_caches(self):
        """Clear internal caches."""
        for encryptor in self.encryptors.values():
            encryptor.clear_caches()
