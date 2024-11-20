import torch
from sphericaltokenizer import CrossModalManager

def demonstrate_spherical_tokenization():
    """
    Demonstrates how spherical tokenization works by showing:
    1. How vectors are transformed in different regions of the embedding space
    2. How the spheroid decomposition affects the transformation
    3. How the momentum-based encryption maintains semantic relationships
    """
    # Initialize with single modality for demonstration
    manager = CrossModalManager(
        modalities={'embedding': 4},  # Using 4D for visualization
        master_key=b'demo-key'
    )

    # Create a set of related vectors in 4D space
    vectors = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Base vector
        [0.9, 0.1, 0.0, 0.0],  # Close to base
        [0.7, 0.3, 0.0, 0.0],  # Moderately far
        [0.0, 1.0, 0.0, 0.0],  # Orthogonal
    ])

    print("\nOriginal vectors:")
    print_vector_relationships(vectors)

    # Transform the vectors
    embeddings = {'embedding': vectors}
    transformed = manager.secure_cross_modal_transform(embeddings)
    transformed_vectors = transformed['embedding']

    print("\nTransformed vectors:")
    print_vector_relationships(transformed_vectors)

    # Demonstrate reversibility
    decrypted = manager.encryptors['embedding'].decrypt_vector(
        transformed_vectors,
        manager.modalities['embedding'].spheroids
    )

    print("\nDecrypted vectors:")
    print_vector_relationships(decrypted)

    # Show how transformation varies by region
    print("\nTransformation effect by region:")
    for i, spheroid in enumerate(manager.modalities['embedding'].spheroids):
        in_spheroid = manager.encryptors['embedding']._vector_in_spheroid(vectors, spheroid)
        print(f"\nSpheroid {i}:")
        print(f"Vectors in this region: {in_spheroid.tolist()}")

def print_vector_relationships(vectors):
    """Print cosine similarities between vectors to show relationship preservation."""
    n = len(vectors)
    similarities = torch.zeros((n, n))
    
    # Compute cosine similarities
    for i in range(n):
        for j in range(n):
            sim = torch.nn.functional.cosine_similarity(
                vectors[i].unsqueeze(0),
                vectors[j].unsqueeze(0)
            )
            similarities[i, j] = sim.item()
    
    print("Cosine similarities:")
    for i in range(n):
        for j in range(n):
            print(f"{similarities[i,j]:6.3f}", end=" ")
        print()

def main():
    print("Spherical Tokenization Demonstration")
    print("===================================")
    demonstrate_spherical_tokenization()

if __name__ == "__main__":
    main()
