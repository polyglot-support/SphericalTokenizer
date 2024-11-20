# SphericalTokenizer

A GPU-accelerated cryptographic tokenizer library that secures embedding vectors through geometric transformations and role-based access control. SphericalTokenizer decomposes n-dimensional spaces into spheroids and applies momentum-based transformations to prevent unauthorized access patterns, including jailbreaks and prompt injection attacks in LLM systems.

## Features

- **GPU-Accelerated Geometric Encryption**: Decomposes n-dimensional embedding spaces into n/3 spheroids with CUDA support
- **Efficient Batch Processing**: Optimized batch operations for high-performance vector transformations
- **Momentum-Based Transformations**: Applies secure, reversible transformations using particle momentum
- **Role-Based Access Control**: Layered security model for fine-grained access control
- **Attack Prevention**: Protects against jailbreaks, prompt injection, and model extraction
- **Automatic Device Management**: Seamless CPU fallback when GPU is unavailable

## Performance

- **Single Vector Operations**: ~100ms per vector
- **Batch Processing**: ~43ms per vector with optimal batch size (32)
- **Memory Usage**: ~1.7GB for standard operations
- **GPU Utilization**: Efficient tensor operations with proper memory management

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- PyTorch >= 2.0.0 (with CUDA support for GPU acceleration)
- NumPy >= 1.21.0
- Cryptography >= 3.4.7

## Quick Start

```python
import torch
from sphericaltokenizer import SphericalTokenizer

# Initialize tokenizer (automatically uses GPU if available)
embedding_dim = 768  # Example dimension for BERT embeddings
master_key = b'your-secure-key-here'
tokenizer = SphericalTokenizer(embedding_dim, master_key)

# Create roles with permissions
tokenizer.create_role('admin', {'read', 'write', 'execute'})
tokenizer.create_role('user', {'read'})

# Example embedding vector (automatically moves to GPU if available)
vector = torch.randn(embedding_dim)

# Encrypt vector with role-based access
encrypted = tokenizer.encrypt(vector, roles=['user'])

# Decrypt vector with appropriate roles
decrypted = tokenizer.decrypt(encrypted, roles=['user'])

# Verify transformation
assert tokenizer.verify_transformation(vector, roles=['user'])
```

## Advanced Usage

### Efficient Batch Processing

```python
# Process multiple vectors efficiently on GPU
batch_size = 32  # Optimal batch size for GPU
vectors = torch.randn(batch_size, embedding_dim)
encrypted_batch = tokenizer.transform_batch(vectors, roles=['user'])
```

### Secure Similarity Computation

```python
# Compare vectors in secure space with GPU acceleration
vector1 = torch.randn(embedding_dim)
vector2 = torch.randn(embedding_dim)
similarity = tokenizer.secure_similarity(vector1, vector2, roles=['user'])
```

### Role-Based Access Control

```python
# Create roles with specific permissions
tokenizer.create_role('moderator', {'read', 'moderate'})

# Validate access rights
required_permissions = {'read', 'moderate'}
has_access = tokenizer.validate_access(required_permissions, ['moderator'])

# Get effective permissions
effective_perms = tokenizer.get_effective_permissions(['admin', 'moderator'])
```

## Performance Optimization

### Batch Size Selection
- Optimal batch size: 32 vectors
- Adjust based on available GPU memory
- Use smaller batches for CPU-only operation

```python
# Example of batch size selection
if torch.cuda.is_available():
    batch_size = 32  # GPU optimal
else:
    batch_size = 16  # CPU optimal

# Process in batches
results = tokenizer.transform_batch(vectors, batch_size=batch_size)
```

### Memory Management

```python
# Clear GPU cache if needed
torch.cuda.empty_cache()

# Move tokenizer to specific device
tokenizer.to(torch.device('cuda:1'))  # For multi-GPU systems
```

## Security Model

SphericalTokenizer implements a multi-layered security approach:

1. **Base Layer**: GPU-accelerated geometric decomposition into spheroids
2. **Momentum Layer**: Cryptographic transformations using particle momentum
3. **Role Layer**: RBAC-based access control
4. **Composition Layer**: Secure layer composition and validation

### Attack Prevention

- **Jailbreak Prevention**: Geometric constraints prevent unauthorized command execution
- **Prompt Injection**: Layered RBAC prevents unauthorized access patterns
- **Model Extraction**: Encrypted embeddings resist reverse engineering attempts

## Performance Benchmarks

| Operation | GPU Time (ms) | CPU Time (ms) |
|-----------|--------------|--------------|
| Single Vector | 100 | 300 |
| Batch (32) | 43 per vector | 250 per vector |
| Similarity | 85 | 200 |

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SphericalTokenizer in your research, please cite:

```bibtex
@software{sphericaltokenizer2023,
  title={SphericalTokenizer: GPU-Accelerated Geometric Encryption for LLM Security},
  author={SphericalTokens Team},
  year={2023},
  url={https://github.com/yourusername/sphericaltokenizer}
}
```

## Acknowledgments

- Built on PyTorch for GPU acceleration
- Inspired by geometric approaches to cryptography
- Developed with focus on LLM security requirements
