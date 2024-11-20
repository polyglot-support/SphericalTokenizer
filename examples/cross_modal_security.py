import torch
import numpy as np
from sphericaltokenizer import SphericalTokenizer
from sphericaltokenizer.cross_modal import CrossModalManager

def simulate_normal_embeddings(batch_size: int, device: torch.device):
    """Simulate normal embeddings for demonstration."""
    return {
        'image': torch.randn(batch_size, 512, dtype=torch.float32, device=device),
        'text': torch.randn(batch_size, 768, dtype=torch.float32, device=device)
    }

def simulate_attack_embeddings(batch_size: int, device: torch.device):
    """Simulate adversarial embeddings that attempt to spoof text patterns."""
    # Create image embeddings that try to mimic text patterns
    attack_images = torch.randn(batch_size, 512, dtype=torch.float32, device=device)
    # Add structured patterns that might try to trigger text behavior
    attack_images = torch.tanh(attack_images * 3.0)  # Concentrate values
    return {
        'image': attack_images,
        'text': torch.randn(batch_size, 768, dtype=torch.float32, device=device)
    }

def main():
    # Initialize cross-modal security
    print("Initializing cross-modal security...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    modalities = {
        'image': 512,  # Common image embedding dimension
        'text': 768    # Common text embedding dimension
    }
    master_key = b'cross-modal-security-key-2023'
    
    manager = CrossModalManager(modalities, master_key, device)
    
    # Process normal embeddings
    print("\nProcessing normal embeddings...")
    normal_embeddings = simulate_normal_embeddings(32, device)
    
    # Transform and check normal embeddings
    transformed_normal = manager.secure_cross_modal_transform(normal_embeddings)
    anomalies_normal = manager.detect_anomalies(normal_embeddings)
    
    print("Normal Embeddings Analysis:")
    for modality in normal_embeddings:
        anomaly_rate = torch.mean(anomalies_normal[modality].float()).item()
        print(f"{modality.title()} Anomaly Rate: {anomaly_rate:.2%}")
    
    consistency = manager.verify_cross_modal_consistency(normal_embeddings)
    print(f"Cross-Modal Consistency: {consistency}")
    
    # Process simulated attack embeddings
    print("\nProcessing potential attack embeddings...")
    attack_embeddings = simulate_attack_embeddings(32, device)
    
    # Analyze attack embeddings
    anomalies_attack = manager.detect_anomalies(attack_embeddings)
    
    print("Attack Embeddings Analysis:")
    for modality in attack_embeddings:
        anomaly_rate = torch.mean(anomalies_attack[modality].float()).item()
        print(f"{modality.title()} Anomaly Rate: {anomaly_rate:.2%}")
    
    consistency = manager.verify_cross_modal_consistency(attack_embeddings)
    print(f"Cross-Modal Consistency: {consistency}")
    
    # Demonstrate entropy analysis
    print("\nEntropy Analysis:")
    for modality in normal_embeddings:
        normal_entropy = torch.mean(manager._compute_entropy_batch(normal_embeddings[modality])).item()
        attack_entropy = torch.mean(manager._compute_entropy_batch(attack_embeddings[modality])).item()
        print(f"{modality.title()}:")
        print(f"  Normal Entropy: {normal_entropy:.4f}")
        print(f"  Attack Entropy: {attack_entropy:.4f}")
        
        config = manager.modalities[modality]
        print(f"  Entropy Thresholds: [{config.entropy_threshold_low:.4f}, {config.entropy_threshold_high:.4f}]")
    
    # Demonstrate transformation consistency
    print("\nTransformation Consistency Analysis:")
    for modality in normal_embeddings:
        config = manager.modalities[modality]
        
        normal_inconsistencies = manager._check_transformation_consistency(
            normal_embeddings[modality],
            config.spheroids
        )
        attack_inconsistencies = manager._check_transformation_consistency(
            attack_embeddings[modality],
            config.spheroids
        )
        
        normal_rate = torch.mean(normal_inconsistencies.float()).item()
        attack_rate = torch.mean(attack_inconsistencies.float()).item()
        
        print(f"{modality.title()}:")
        print(f"  Normal Inconsistency Rate: {normal_rate:.2%}")
        print(f"  Attack Inconsistency Rate: {attack_rate:.2%}")
    
    # Memory usage statistics
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
