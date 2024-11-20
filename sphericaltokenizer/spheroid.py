import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class Spheroid:
    """
    Represents a spheroid in n-dimensional space.
    
    Attributes:
        center: The center point of the spheroid
        radius: float
        axes: Principal axes of the spheroid
        eccentricity: Eccentricity factor for non-uniform scaling
    """
    center: torch.Tensor
    radius: float
    axes: torch.Tensor
    eccentricity: float

class SpheroidGenerator:
    """
    Handles the decomposition of n-dimensional embedding spaces into spheroids.
    """
    
    def __init__(self, embedding_dim: int, num_spheroids: Optional[int] = None):
        """
        Initialize the SpheroidGenerator.
        
        Args:
            embedding_dim: Dimension of the embedding space
            num_spheroids: Optional override for number of spheroids
        """
        self.embedding_dim = embedding_dim
        self.num_spheroids = num_spheroids or max(1, embedding_dim // 3)
        
        # Validate dimensions
        if embedding_dim < 3:
            raise ValueError("Embedding dimension must be at least 3")
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize random state and cache
        self._cache: Dict[int, torch.Tensor] = {}
        
        # Precompute common values
        self._precompute_values()
    
    def _precompute_values(self):
        """Precompute commonly used values."""
        # Precompute angles for center generation
        self.phi = (1 + torch.sqrt(torch.tensor(5.0, device=self.device))) / 2
        self.angle_indices = torch.arange(self.embedding_dim // 2, device=self.device)
        self.base_angles = 2 * torch.pi * self.phi * self.angle_indices
        
        # Precompute indices for efficient operations
        self.even_indices = torch.arange(0, self.embedding_dim, 2, device=self.device)
        self.odd_indices = torch.arange(1, self.embedding_dim, 2, device=self.device)
        
        # Precompute eccentricity factors
        indices = torch.arange(self.num_spheroids, device=self.device, dtype=torch.float32)
        self.eccentricity_factors = 1.0 + 0.5 * torch.sin(
            2 * torch.pi * indices / self.num_spheroids
        )
    
    def generate_spheroids(self) -> List[Spheroid]:
        """
        Generate a set of spheroids that optimally cover the embedding space.
        
        Returns:
            List of Spheroid objects positioned in the embedding space
        """
        spheroids = []
        
        # First spheroid at origin for guaranteed coverage
        center = torch.zeros(self.embedding_dim, device=self.device)
        radius = 1.0
        axes = self._generate_axes(0)
        spheroids.append(Spheroid(
            center=center,
            radius=radius,
            axes=axes,
            eccentricity=self.eccentricity_factors[0].item()
        ))
        
        # Generate remaining spheroids in parallel
        centers = self._generate_centers_batch(
            torch.arange(1, self.num_spheroids, device=self.device)
        )
        
        for i, center in enumerate(centers, 1):
            radius = self._calculate_radius(center, spheroids)
            axes = self._generate_axes(i)
            spheroids.append(Spheroid(
                center=center,
                radius=radius,
                axes=axes,
                eccentricity=self.eccentricity_factors[i].item()
            ))
        
        return spheroids
    
    def _generate_centers_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Generate multiple centers efficiently in a batch.
        
        Args:
            indices: Array of spheroid indices
        
        Returns:
            Array of center coordinates
        """
        t = indices.float() / self.num_spheroids
        angles = self.base_angles.unsqueeze(0) * t.unsqueeze(1)
        
        # Initialize centers tensor
        centers = torch.zeros(
            (len(indices), self.embedding_dim),
            device=self.device
        )
        
        # Compute all cosines and sines at once
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # Fill even indices with cosines
        centers[:, self.even_indices[:len(self.angle_indices)]] = cos_vals * (1 - t.unsqueeze(1))
        
        # Fill odd indices with sines where they exist
        valid_odd = self.odd_indices[self.odd_indices < self.embedding_dim]
        if len(valid_odd) > 0:
            centers[:, valid_odd] = sin_vals[:, :len(valid_odd)] * (1 - t.unsqueeze(1))
        
        # Scale centers for better coverage
        return centers * 2.0
    
    @lru_cache(maxsize=128)
    def _generate_axes(self, index: int) -> torch.Tensor:
        """
        Generate orthogonal axes for the spheroid using cached QR decomposition.
        
        Args:
            index: Spheroid index for cache key
        
        Returns:
            Matrix of orthogonal axes
        """
        if index not in self._cache:
            # Generate random matrix
            random_matrix = torch.randn(
                (self.embedding_dim, self.embedding_dim),
                device=self.device
            )
            
            # Use QR decomposition to get orthogonal axes
            Q, _ = torch.linalg.qr(random_matrix)
            
            # Ensure Q is properly normalized
            norms = torch.norm(Q, dim=0)
            Q = Q / norms
            
            self._cache[index] = Q
        
        return self._cache[index]
    
    def _calculate_radius(self, center: torch.Tensor, existing_spheroids: List[Spheroid]) -> float:
        """
        Calculate optimal radius for a new spheroid based on existing spheroids.
        
        Args:
            center: Center coordinates of the new spheroid
            existing_spheroids: List of already placed spheroids
        
        Returns:
            Optimal radius value
        """
        if not existing_spheroids:
            return 1.0
        
        # Calculate distances to all existing spheroid centers at once
        centers = torch.stack([s.center for s in existing_spheroids])
        distances = torch.norm(center - centers, dim=1)
        
        # Find minimum distance and calculate radius
        min_distance = torch.min(distances).item()
        return min_distance * 0.6
    
    def contains_point(self, spheroid: Spheroid, point: torch.Tensor) -> bool:
        """
        Check if a point lies within a spheroid.
        
        Args:
            spheroid: Spheroid object to check against
            point: Point coordinates to test
        
        Returns:
            Boolean indicating whether the point is inside the spheroid
        """
        # Transform and scale in one operation
        relative_point = point - spheroid.center
        transformed_scaled = torch.mv(spheroid.axes.T, relative_point) / (
            spheroid.radius * spheroid.eccentricity
        )
        
        # Check if point lies within spheroid
        return torch.sum(transformed_scaled ** 2) <= 1.0
    
    def get_containing_spheroids(self, point: torch.Tensor, spheroids: List[Spheroid]) -> List[int]:
        """
        Find all spheroids that contain a given point.
        
        Args:
            point: Point coordinates to test
            spheroids: List of spheroids to check against
        
        Returns:
            List of indices of containing spheroids
        """
        # Vectorized containment check
        return [i for i, spheroid in enumerate(spheroids) 
                if self.contains_point(spheroid, point)]
    
    def to_device(self, device: torch.device):
        """
        Move the generator to specified device.
        
        Args:
            device: Target device (cuda or cpu)
        """
        self.device = device
        self._cache.clear()
        self._precompute_values()
        
    def numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor on current device."""
        return torch.from_numpy(array).to(self.device)
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
