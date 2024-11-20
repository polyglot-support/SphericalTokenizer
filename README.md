# SphericalTokenizer

A secure tokenization system that uses spherical geometry and particle momentum simulation to create RBAC-secured transformations for embedding vectors. This system provides protection against jailbreaks and other attacks on LLM systems by creating secure, reversible transformations that maintain semantic relationships while preventing unauthorized access.

## Features

- Spheroid-based decomposition of embedding space
- Momentum-based encryption with configurable complexity levels
- Cross-modal security with consistency verification
- Anomaly detection for embedding manipulation attempts
- Device-independent operation (CPU/GPU support)
- Batch processing support
- Robust numerical stability

## Installation

```bash
pip install sphericaltokenizer
```

## Quick Start

```python
import torch
from sphericaltokenizer import CrossModalManager

# Initialize the manager with desired complexity
manager = CrossModalManager(
    modalities={
        'image': 512,
        'text': 768
    },
    master_key=b'your-secure-key',
    complexity='standard'  # Choose: 'minimal', 'basic', 'standard', 'high'
)

# Transform embeddings
embeddings = {
    'image': image_embedding,  # Your image embedding tensor
    'text': text_embedding     # Your text embedding tensor
}

transformed = manager.secure_cross_modal_transform(embeddings)
```

## Complexity Levels

SphericalTokenizer supports different complexity levels to balance security and performance:

1. **Minimal** (`complexity='minimal'`)
   - Simple static shift transformation
   - Fastest performance
   - Suitable for basic context differentiation
   - Minimal computational overhead

2. **Basic** (`complexity='basic'`)
   - Basic rotation and scaling
   - Good performance
   - Suitable for low-security applications
   - Light computational overhead

3. **Standard** (`complexity='standard'`)
   - Full momentum-based transformation
   - Balanced performance
   - Suitable for most security needs
   - Moderate computational overhead

4. **High** (`complexity='high'`)
   - Enhanced security with additional transformations
   - Higher computational cost
   - Suitable for high-security applications
   - Maximum protection

See [examples/complexity_levels.py](examples/complexity_levels.py) for a demonstration of each level.

## Examples

### Complexity Levels Demo

See [examples/complexity_levels.py](examples/complexity_levels.py) for a demonstration of:
- Different transformation complexities
- Performance benchmarks
- Security-performance tradeoffs
- Effect on vector properties

### Basic Spherical Tokenization

See [examples/spherical_tokenization.py](examples/spherical_tokenization.py) for a demonstration of:
- Vector transformation in different regions
- Spheroid decomposition effects
- Semantic relationship preservation
- Transformation reversibility

### Cross-Modal Security

See [examples/cross_modal_security.py](examples/cross_modal_security.py) for an example of:
- Cross-modal transformation security
- Anomaly detection
- Consistency verification
- Device movement handling

## Documentation

For detailed design information and implementation details, see [DESIGN.md](DESIGN.md).

### Key Components

1. **CrossModalManager**: Main interface for secure transformations
   - Manages modality-specific configurations
   - Handles device movement
   - Provides anomaly detection
   - Ensures cross-modal consistency

2. **Momentum-Based Encryption**
   - Configurable complexity levels
   - Deterministic transformations
   - Reversible operations
   - Scale-invariant transformations

3. **Spheroid Decomposition**
   - Local transformation regions
   - Smooth transitions
   - Configurable containment
   - Device-independent operations

## Security Features

- Value range validation (-100 to 100)
- Magnitude checks (1e-8 to 1e4)
- NaN/Inf detection
- Entropy-based anomaly detection
- Reconstruction error validation
- Transformation effect verification
- Cross-modal consistency checks

## Usage Guidelines

### Initialization

```python
manager = CrossModalManager(
    modalities={'image': 512, 'text': 768},
    master_key=b'your-secret-key',
    complexity='standard'  # Choose complexity level
)
```

### Transformation

```python
transformed = manager.secure_cross_modal_transform({
    'image': image_embedding,
    'text': text_embedding
})
```

### Validation

```python
anomalies = manager.detect_anomalies(embeddings)
is_consistent = manager.verify_cross_modal_consistency(embeddings)
```

### Device Movement

```python
# Move to GPU
manager.to(torch.device('cuda'))

# Move back to CPU
manager.to(torch.device('cpu'))
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this in your research, please cite:

```bibtex
@software{sphericaltokenizer2024,
  title={SphericalTokenizer: GPU-Accelerated Geometric Encryption for LLM Security},
  author={SphericalTokens Team},
  year={2024},
  url={https://github.com/polyglot-support/sphericaltokenizer}
}
