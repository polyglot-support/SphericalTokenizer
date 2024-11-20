import pytest
import torch
import time
import psutil
import os
from sphericaltokenizer import CrossModalManager
from tabulate import tabulate
import sys

def measure_performance(manager, embeddings, num_iterations=100):
    """Measure performance of transformations."""
    # Warm up
    for _ in range(10):
        manager.secure_cross_modal_transform(embeddings)
    
    # Measure
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        manager.secure_cross_modal_transform(embeddings)
    elapsed_time = (time.perf_counter() - start_time) * 1000 / num_iterations  # ms per iteration
    
    return elapsed_time

def get_process_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

@pytest.mark.benchmark
class TestComplexityPerformance:
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings of different sizes."""
        sizes = {
            'small': 64,    # Small embeddings
            'medium': 256,  # Medium embeddings
            'large': 512    # Large embeddings
        }
        
        embeddings = {}
        for name, size in sizes.items():
            embeddings[name] = {
                'test': torch.randn(1, size)  # Add batch dimension
            }
        return embeddings
    
    def test_complexity_performance_comparison(self, sample_embeddings, capsys):
        """Compare performance across complexity levels."""
        complexities = ['minimal', 'basic', 'standard', 'high']
        benchmark_data = []
        
        sys.stdout.write("\nComplexity Performance Benchmark\n")
        sys.stdout.write("==============================\n")
        sys.stdout.flush()
        
        for size, embeddings in sample_embeddings.items():
            for complexity in complexities:
                # Initialize manager with specific complexity
                manager = CrossModalManager(
                    modalities={'test': embeddings['test'].shape[1]},
                    master_key=b'test-key',
                    complexity=complexity
                )
                
                # Measure performance
                elapsed_time = measure_performance(manager, embeddings)
                
                # Add to benchmark data
                benchmark_data.append([
                    size.title(),
                    str(embeddings['test'].shape[1]),
                    complexity.title(),
                    f"{elapsed_time:.3f}"
                ])
        
        # Display results in table format
        headers = ["Size", "Dimensions", "Complexity", "Time (ms/iter)"]
        table = tabulate(benchmark_data, headers=headers, tablefmt="grid")
        sys.stdout.write("\n" + table + "\n")
        sys.stdout.flush()
    
    def test_batch_size_scaling(self, capsys):
        """Test how different complexities scale with batch size."""
        dim = 256
        batch_sizes = [1, 4, 8, 16]
        complexities = ['minimal', 'basic', 'standard', 'high']
        benchmark_data = []
        
        sys.stdout.write("\nBatch Size Scaling Benchmark\n")
        sys.stdout.write("==========================\n")
        sys.stdout.flush()
        
        for complexity in complexities:
            for batch_size in batch_sizes:
                # Create batch
                embeddings = {
                    'test': torch.randn(batch_size, dim)
                }
                
                # Initialize manager
                manager = CrossModalManager(
                    modalities={'test': dim},
                    master_key=b'test-key',
                    complexity=complexity
                )
                
                # Measure performance
                elapsed_time = measure_performance(manager, embeddings)
                
                # Add to benchmark data
                benchmark_data.append([
                    complexity.title(),
                    str(batch_size),
                    f"{elapsed_time:.3f}",
                    f"{elapsed_time/batch_size:.3f}"
                ])
        
        # Display results in table format
        headers = ["Complexity", "Batch Size", "Total Time (ms)", "Time per Sample (ms)"]
        table = tabulate(benchmark_data, headers=headers, tablefmt="grid")
        sys.stdout.write("\n" + table + "\n")
        sys.stdout.flush()
    
    def test_memory_usage(self, capsys):
        """Test memory usage of different complexity levels."""
        dim = 256
        batch_size = 8
        complexities = ['minimal', 'basic', 'standard', 'high']
        benchmark_data = []
        
        sys.stdout.write("\nMemory Usage Benchmark\n")
        sys.stdout.write("====================\n")
        sys.stdout.flush()
        
        # Get baseline memory usage
        baseline_memory = get_process_memory()
        
        for complexity in complexities:
            # Clear any previous data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Create embeddings
            embeddings = {
                'test': torch.randn(batch_size, dim)
            }
            
            # Initialize manager
            manager = CrossModalManager(
                modalities={'test': dim},
                master_key=b'test-key',
                complexity=complexity
            )
            
            # Measure memory before operation
            pre_memory = get_process_memory()
            
            # Run operations
            for _ in range(10):  # Multiple iterations to ensure memory is allocated
                _ = manager.secure_cross_modal_transform(embeddings)
            
            # Measure memory after operation
            post_memory = get_process_memory()
            
            # Calculate memory usage
            memory_used = post_memory - baseline_memory
            
            # Add to benchmark data
            benchmark_data.append([
                complexity.title(),
                f"{memory_used:.2f}"
            ])
        
        # Display results in table format
        headers = ["Complexity", "Memory Usage (MB)"]
        table = tabulate(benchmark_data, headers=headers, tablefmt="grid")
        sys.stdout.write("\n" + table + "\n")
        sys.stdout.flush()
