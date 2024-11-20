import numpy as np
import torch
import time
from sphericaltokenizer import SphericalTokenizer

def benchmark_operation(name, operation, *args, **kwargs):
    """Utility function to benchmark operations."""
    # Ensure GPU sync before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    result = operation(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"{name}: {(end_time - start_time)*1000:.2f} ms")
    return result

def print_gpu_stats():
    """Print GPU memory statistics."""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Initialize tokenizer
    print("Initializing SphericalTokenizer...")
    embedding_dim = 768  # Standard BERT dimension
    master_key = b'example-master-key-2023'
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = SphericalTokenizer(embedding_dim, master_key, device=device)

    # Create roles with different permission levels
    print("\nCreating roles...")
    tokenizer.create_role('admin', {'read', 'write', 'execute'})
    tokenizer.create_role('user', {'read'})

    # Generate example vectors
    print("\nGenerating example vectors...")
    original_vector = torch.randn(embedding_dim, device=device)
    print(f"Original vector norm: {torch.norm(original_vector):.4f}")

    # Benchmark basic encryption/decryption
    print("\nBenchmarking basic encryption/decryption...")
    encrypted = benchmark_operation(
        "Encryption",
        tokenizer.encrypt,
        original_vector,
        roles=['user']
    )
    decrypted = benchmark_operation(
        "Decryption",
        tokenizer.decrypt,
        encrypted,
        roles=['user']
    )
    
    similarity = torch.dot(original_vector, decrypted) / (
        torch.norm(original_vector) * torch.norm(decrypted)
    )
    print(f"Original vs Decrypted Similarity: {similarity:.6f}")

    # Clear cache before permission checks
    clear_gpu_cache()
    print_gpu_stats()

    # Benchmark role-based access control
    print("\nBenchmarking role-based access control...")
    required_permissions = {'read', 'write'}
    user_access = benchmark_operation(
        "User permission check",
        tokenizer.validate_access,
        required_permissions,
        ['user']
    )
    admin_access = benchmark_operation(
        "Admin permission check",
        tokenizer.validate_access,
        required_permissions,
        ['admin']
    )
    print(f"User has required permissions: {user_access}")
    print(f"Admin has required permissions: {admin_access}")

    # Clear cache before similarity computation
    clear_gpu_cache()
    print_gpu_stats()

    # Benchmark secure similarity computation
    print("\nBenchmarking secure similarity computation...")
    vector1 = torch.randn(embedding_dim, device=device)
    vector2 = torch.randn(embedding_dim, device=device)
    
    # Compare vectors in original and secure space
    original_sim = torch.dot(vector1, vector2) / (
        torch.norm(vector1) * torch.norm(vector2)
    )
    secure_sim = benchmark_operation(
        "Secure similarity",
        tokenizer.secure_similarity,
        vector1,
        vector2
    )
    print(f"Original similarity: {original_sim:.6f}")
    print(f"Secure similarity: {secure_sim:.6f}")

    # Clear cache before batch processing
    clear_gpu_cache()
    print_gpu_stats()

    # Benchmark batch processing with optimal batch sizes
    print("\nBenchmarking batch processing...")
    batch_sizes = [16, 32, 48]  # Smaller batch sizes for better GPU utilization
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        vectors = torch.randn(batch_size, embedding_dim, device=device)
        
        encrypted_batch = benchmark_operation(
            f"Batch encryption ({batch_size} vectors)",
            tokenizer.transform_batch,
            vectors,
            roles=['user']
        )
        
        decrypted_batch = benchmark_operation(
            f"Batch decryption ({batch_size} vectors)",
            tokenizer.transform_batch,
            encrypted_batch,
            roles=['user'],
            decrypt=True
        )
        
        # Verify batch processing accuracy
        accuracy = torch.mean(torch.all(
            torch.isclose(vectors, decrypted_batch, atol=1e-6),
            dim=1
        ).float())
        print(f"Batch processing accuracy: {accuracy * 100:.2f}%")
        
        # Benchmark batch similarity
        vectors1 = torch.randn(batch_size, embedding_dim, device=device)
        vectors2 = torch.randn(batch_size, embedding_dim, device=device)
        
        similarities = benchmark_operation(
            f"Batch secure similarity ({batch_size} pairs)",
            tokenizer.batch_secure_similarity,
            vectors1,
            vectors2
        )
        print(f"Average similarity: {torch.mean(similarities):.6f}")
        
        # Clear cache after each batch
        clear_gpu_cache()
        print_gpu_stats()

    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
