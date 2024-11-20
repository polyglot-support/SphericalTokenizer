import torch
from sphericaltokenizer import CrossModalManager

def demonstrate_complexity_levels():
    """
    Demonstrates different complexity levels of transformation:
    - minimal: Simple static shift (fastest, lowest security)
    - basic: Basic rotation and scaling
    - standard: Full momentum-based transformation
    - high: Enhanced security with additional transformations
    """
    # Create sample embedding
    embedding_dim = 4  # Using 4D for easy visualization
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embeddings = {'test': vector}

    # Test each complexity level
    complexity_levels = ['minimal', 'basic', 'standard', 'high']
    
    print("\nTransformation Effects by Complexity Level")
    print("=========================================")
    
    for complexity in complexity_levels:
        # Initialize manager with specific complexity
        manager = CrossModalManager(
            modalities={'test': embedding_dim},
            master_key=b'demo-key',
            complexity=complexity
        )
        
        # Transform the vector
        transformed = manager.secure_cross_modal_transform(embeddings)
        transformed_vector = transformed['test']
        
        # Calculate change metrics
        magnitude_change = torch.norm(transformed_vector - vector).item()
        direction_change = torch.nn.functional.cosine_similarity(
            transformed_vector.unsqueeze(0),
            vector.unsqueeze(0)
        ).item()
        
        print(f"\nComplexity Level: {complexity}")
        print(f"Original vector:     {vector.tolist()}")
        print(f"Transformed vector:  {transformed_vector.tolist()}")
        print(f"Magnitude change:    {magnitude_change:.3f}")
        print(f"Direction change:    {1 - direction_change:.3f}")  # 0 means same direction
        
        # Performance benchmark
        if torch.cuda.is_available():
            # Warm up
            for _ in range(100):
                manager.secure_cross_modal_transform(embeddings)
            
            # Measure
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(1000):
                manager.secure_cross_modal_transform(embeddings)
            end.record()
            
            torch.cuda.synchronize()
            print(f"Average time (ms):   {start.elapsed_time(end) / 1000:.3f}")

def main():
    print("SphericalTokenizer Complexity Levels Demo")
    print("========================================")
    demonstrate_complexity_levels()

if __name__ == "__main__":
    main()
