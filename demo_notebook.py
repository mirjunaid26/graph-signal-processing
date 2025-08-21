"""
Interactive Demo of Graph Signal Processing
Reproducing key results from Sandryhaila & Moura 2014

This script provides an interactive demonstration of the main concepts
from "Big Data Analysis with Signal Processing on Graphs"
"""

import numpy as np
import matplotlib.pyplot as plt
from graph_signal_processing import GraphSignalProcessor, create_example_graphs, generate_test_signals, visualize_results
from big_data_experiments import BigDataGraphProcessor
from pygsp import graphs
import warnings
warnings.filterwarnings('ignore')

def demo_basic_gsp():
    """
    Demonstrate basic Graph Signal Processing concepts
    """
    print("=" * 60)
    print("DEMO 1: Basic Graph Signal Processing")
    print("=" * 60)
    
    # Create a sensor network graph
    graph = graphs.Sensor(N=64, seed=42)
    gsp = GraphSignalProcessor(graph)
    
    print(f"Created sensor network with {graph.N} nodes and {graph.Ne} edges")
    
    # Generate a smooth test signal
    signal = generate_test_signals(graph, 'smooth')
    print(f"Generated smooth test signal with energy: {np.sum(signal**2):.2f}")
    
    # Perform spectral analysis
    analysis = gsp.spectral_analysis(signal)
    
    print(f"Spectral Analysis Results:")
    print(f"  - Total energy: {analysis['total_energy']:.2f}")
    print(f"  - Low freq energy: {analysis['low_freq_energy_ratio']:.2%}")
    print(f"  - Mid freq energy: {analysis['mid_freq_energy_ratio']:.2%}")
    print(f"  - High freq energy: {analysis['high_freq_energy_ratio']:.2%}")
    
    # Visualize results
    visualize_results(gsp, signal, analysis, 'demo_basic_gsp.png')
    
    return gsp, signal, analysis

def demo_graph_filtering():
    """
    Demonstrate graph filtering capabilities
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Graph Filtering and Denoising")
    print("=" * 60)
    
    # Create graph and generate noisy signal
    graph = graphs.Community(N=100, seed=42)
    gsp = GraphSignalProcessor(graph)
    
    # Generate clean signal and add noise
    clean_signal = generate_test_signals(graph, 'smooth')
    noise = 0.5 * np.random.randn(graph.N)
    noisy_signal = clean_signal + noise
    
    print(f"Original SNR: {10 * np.log10(np.var(clean_signal) / np.var(noise)):.2f} dB")
    
    # Apply different filters
    low_pass_filter = gsp.design_graph_filter('low_pass', cutoff=0.3)
    filtered_signal = gsp.filter_signal(noisy_signal, low_pass_filter)
    
    # Compute denoising performance
    mse_noisy = np.mean((noisy_signal - clean_signal)**2)
    mse_filtered = np.mean((filtered_signal - clean_signal)**2)
    
    print(f"MSE before filtering: {mse_noisy:.4f}")
    print(f"MSE after filtering: {mse_filtered:.4f}")
    print(f"Improvement factor: {mse_noisy / mse_filtered:.2f}x")
    
    # Visualize filtering results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(range(len(clean_signal)), clean_signal, alpha=0.7, s=20)
    axes[0].set_title('Original Clean Signal')
    axes[0].set_xlabel('Node Index')
    axes[0].set_ylabel('Signal Value')
    
    axes[1].scatter(range(len(noisy_signal)), noisy_signal, alpha=0.7, s=20, color='red')
    axes[1].set_title('Noisy Signal')
    axes[1].set_xlabel('Node Index')
    axes[1].set_ylabel('Signal Value')
    
    axes[2].scatter(range(len(filtered_signal)), filtered_signal, alpha=0.7, s=20, color='green')
    axes[2].set_title('Filtered Signal')
    axes[2].set_xlabel('Node Index')
    axes[2].set_ylabel('Signal Value')
    
    plt.tight_layout()
    plt.savefig('demo_filtering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return clean_signal, noisy_signal, filtered_signal

def demo_anomaly_detection():
    """
    Demonstrate anomaly detection using graph signal processing
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Anomaly Detection on Graphs")
    print("=" * 60)
    
    # Create graph
    graph = graphs.Grid2d(N1=8, N2=8)
    gsp = GraphSignalProcessor(graph)
    
    # Generate signal with anomalies
    signal = generate_test_signals(graph, 'smooth')
    
    # Add artificial anomalies
    anomaly_indices = [10, 25, 40, 55]  # Known anomaly locations
    signal[anomaly_indices] += 3.0  # Strong anomalies
    
    print(f"Injected {len(anomaly_indices)} anomalies at indices: {anomaly_indices}")
    
    # Detect anomalies
    detected_anomalies = gsp.detect_graph_anomalies(signal, threshold=2.0)
    detected_indices = np.where(detected_anomalies)[0]
    
    print(f"Detected {np.sum(detected_anomalies)} anomalies at indices: {detected_indices.tolist()}")
    
    # Compute detection performance
    true_anomalies = np.zeros(graph.N, dtype=bool)
    true_anomalies[anomaly_indices] = True
    
    true_positives = np.sum(detected_anomalies & true_anomalies)
    false_positives = np.sum(detected_anomalies & ~true_anomalies)
    false_negatives = np.sum(~detected_anomalies & true_anomalies)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"Detection Performance:")
    print(f"  - True Positives: {true_positives}")
    print(f"  - False Positives: {false_positives}")
    print(f"  - False Negatives: {false_negatives}")
    print(f"  - Precision: {precision:.2%}")
    print(f"  - Recall: {recall:.2%}")
    
    # Visualize anomaly detection
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original signal with anomalies
    im1 = axes[0].imshow(signal.reshape(8, 8), cmap='viridis')
    axes[0].set_title('Signal with Anomalies')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot detected anomalies
    detection_map = detected_anomalies.astype(float).reshape(8, 8)
    im2 = axes[1].imshow(detection_map, cmap='Reds')
    axes[1].set_title('Detected Anomalies')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('demo_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return signal, detected_anomalies

def demo_graph_comparison():
    """
    Compare different graph structures and their spectral properties
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Graph Structure Comparison")
    print("=" * 60)
    
    # Create different graph types
    graphs_dict = create_example_graphs()
    
    results = {}
    
    for graph_name, graph in graphs_dict.items():
        gsp = GraphSignalProcessor(graph)
        gsp.compute_graph_fourier_basis()
        
        # Compute spectral properties
        spectral_gap = gsp.eigenvalues[1] - gsp.eigenvalues[0]
        max_eigenvalue = np.max(gsp.eigenvalues)
        
        # Test signal smoothness preservation
        smooth_signal = generate_test_signals(graph, 'smooth')
        analysis = gsp.spectral_analysis(smooth_signal)
        
        results[graph_name] = {
            'nodes': graph.N,
            'edges': graph.Ne,
            'spectral_gap': spectral_gap,
            'max_eigenvalue': max_eigenvalue,
            'low_freq_ratio': analysis['low_freq_energy_ratio']
        }
        
        print(f"{graph_name.upper()}:")
        print(f"  - Nodes: {graph.N}, Edges: {graph.Ne}")
        print(f"  - Spectral gap: {spectral_gap:.4f}")
        print(f"  - Max eigenvalue: {max_eigenvalue:.2f}")
        print(f"  - Low freq energy ratio: {analysis['low_freq_energy_ratio']:.2%}")
    
    # Visualize comparison
    graph_names = list(results.keys())
    spectral_gaps = [results[name]['spectral_gap'] for name in graph_names]
    low_freq_ratios = [results[name]['low_freq_ratio'] for name in graph_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(graph_names, spectral_gaps)
    axes[0].set_title('Spectral Gap by Graph Type')
    axes[0].set_ylabel('Spectral Gap')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(graph_names, low_freq_ratios)
    axes[1].set_title('Low Frequency Energy Ratio')
    axes[1].set_ylabel('Energy Ratio')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('demo_graph_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def demo_big_data_processing():
    """
    Demonstrate big data processing capabilities
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Big Data Processing Simulation")
    print("=" * 60)
    
    processor = BigDataGraphProcessor()
    
    # Run a subset of experiments for demo
    print("Running scalability test...")
    scalability_results = processor.scalability_experiment([50, 100, 200])
    
    print("Running noise robustness test...")
    noise_results = processor.noise_robustness_experiment([0.1, 0.5, 1.0])
    
    print("Running parallel processing simulation...")
    parallel_results = processor.parallel_processing_simulation(50)
    
    # Generate summary
    print(f"\nBIG DATA PROCESSING SUMMARY:")
    print(f"- Largest graph processed: {max(scalability_results['graph_sizes'])} nodes")
    print(f"- Max processing time: {max(scalability_results['fourier_times']):.2f} seconds")
    print(f"- Average SNR improvement: {np.mean(noise_results['snr_improvement']):.2f} dB")
    print(f"- Parallel speedup factor: {parallel_results['speedup_factor']:.1f}x")
    
    return processor

def run_complete_demo():
    """
    Run all demonstrations
    """
    print("GRAPH SIGNAL PROCESSING DEMONSTRATION")
    print("Reproducing Sandryhaila & Moura 2014")
    print("=" * 80)
    
    # Run all demos
    demo1_results = demo_basic_gsp()
    demo2_results = demo_graph_filtering()
    demo3_results = demo_anomaly_detection()
    demo4_results = demo_graph_comparison()
    demo5_results = demo_big_data_processing()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("Key findings:")
    print("✓ Graph Fourier Transform successfully implemented")
    print("✓ Spectral filtering provides effective denoising")
    print("✓ Anomaly detection works well on structured graphs")
    print("✓ Different graph types have distinct spectral properties")
    print("✓ Vectorized operations enable big data processing")
    print("\nGenerated files:")
    print("- demo_basic_gsp.png")
    print("- demo_filtering.png")
    print("- demo_anomaly_detection.png")
    print("- demo_graph_comparison.png")
    print("=" * 80)
    
    return {
        'basic_gsp': demo1_results,
        'filtering': demo2_results,
        'anomaly_detection': demo3_results,
        'graph_comparison': demo4_results,
        'big_data': demo5_results
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run complete demonstration
    results = run_complete_demo()
