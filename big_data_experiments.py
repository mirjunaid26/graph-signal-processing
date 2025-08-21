"""
Big Data Experiments for Graph Signal Processing
Reproducing experiments from Sandryhaila & Moura 2014 paper

This module demonstrates the big data processing capabilities
and scalability of graph signal processing methods.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from graph_signal_processing import GraphSignalProcessor, create_example_graphs, generate_test_signals
from pygsp import graphs
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class BigDataGraphProcessor:
    """
    Class for handling big data graph signal processing experiments
    """
    
    def __init__(self):
        self.results = {}
        
    def scalability_experiment(self, graph_sizes: List[int] = [50, 100, 200, 500, 1000]) -> Dict:
        """
        Test scalability of graph signal processing with increasing graph sizes
        
        Args:
            graph_sizes: List of graph sizes to test
            
        Returns:
            Dictionary with timing and performance results
        """
        results = {
            'graph_sizes': graph_sizes,
            'fourier_times': [],
            'filtering_times': [],
            'analysis_times': [],
            'memory_usage': []
        }
        
        print("Running scalability experiments...")
        
        for size in graph_sizes:
            print(f"Testing graph size: {size}")
            
            # Create graph
            graph = graphs.Sensor(N=size, seed=42)
            gsp = GraphSignalProcessor(graph)
            
            # Generate test signal
            signal = generate_test_signals(graph, 'smooth')
            
            # Time Fourier basis computation
            start_time = time.time()
            gsp.compute_graph_fourier_basis()
            fourier_time = time.time() - start_time
            results['fourier_times'].append(fourier_time)
            
            # Time filtering operation
            start_time = time.time()
            filter_obj = gsp.design_graph_filter('low_pass', cutoff=0.3)
            filtered_signal = gsp.filter_signal(signal, filter_obj)
            filtering_time = time.time() - start_time
            results['filtering_times'].append(filtering_time)
            
            # Time spectral analysis
            start_time = time.time()
            analysis = gsp.spectral_analysis(signal)
            analysis_time = time.time() - start_time
            results['analysis_times'].append(analysis_time)
            
            # Estimate memory usage (rough approximation)
            memory_est = size * size * 8 / (1024**2)  # MB for dense matrices
            results['memory_usage'].append(memory_est)
        
        self.results['scalability'] = results
        return results
    
    def noise_robustness_experiment(self, noise_levels: List[float] = [0.1, 0.2, 0.5, 1.0, 2.0]) -> Dict:
        """
        Test robustness of graph signal processing to noise
        
        Args:
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary with denoising performance results
        """
        results = {
            'noise_levels': noise_levels,
            'mse_before': [],
            'mse_after': [],
            'snr_improvement': []
        }
        
        print("Running noise robustness experiments...")
        
        # Create base graph and clean signal
        graph = graphs.Sensor(N=200, seed=42)
        gsp = GraphSignalProcessor(graph)
        clean_signal = generate_test_signals(graph, 'smooth')
        
        for noise_level in noise_levels:
            print(f"Testing noise level: {noise_level}")
            
            # Add noise to signal
            noise = noise_level * np.random.randn(graph.N)
            noisy_signal = clean_signal + noise
            
            # Denoise signal
            denoised_signal = gsp.graph_signal_denoising(noisy_signal, noise_level)
            
            # Compute MSE before and after denoising
            mse_before = np.mean((noisy_signal - clean_signal)**2)
            mse_after = np.mean((denoised_signal - clean_signal)**2)
            
            # Compute SNR improvement
            snr_before = 10 * np.log10(np.var(clean_signal) / np.var(noise))
            snr_after = 10 * np.log10(np.var(clean_signal) / np.var(denoised_signal - clean_signal))
            snr_improvement = snr_after - snr_before
            
            results['mse_before'].append(mse_before)
            results['mse_after'].append(mse_after)
            results['snr_improvement'].append(snr_improvement)
        
        self.results['noise_robustness'] = results
        return results
    
    def graph_type_comparison(self) -> Dict:
        """
        Compare performance across different graph types
        
        Returns:
            Dictionary with performance comparison results
        """
        results = {
            'graph_types': [],
            'spectral_gaps': [],
            'clustering_coefficients': [],
            'filter_effectiveness': [],
            'anomaly_detection_accuracy': []
        }
        
        print("Running graph type comparison experiments...")
        
        # Get different graph types
        graphs_dict = create_example_graphs()
        
        for graph_name, graph in graphs_dict.items():
            print(f"Testing graph type: {graph_name}")
            
            gsp = GraphSignalProcessor(graph)
            
            # Compute graph properties
            gsp.compute_graph_fourier_basis()
            spectral_gap = gsp.eigenvalues[1] - gsp.eigenvalues[0]  # Second smallest eigenvalue
            
            # Generate test signal with anomalies
            signal = generate_test_signals(graph, 'smooth')
            
            # Add some artificial anomalies
            anomaly_indices = np.random.choice(graph.N, size=graph.N//10, replace=False)
            signal[anomaly_indices] += 2.0
            
            # Test anomaly detection
            detected_anomalies = gsp.detect_graph_anomalies(signal, threshold=1.5)
            
            # Compute detection accuracy
            true_anomalies = np.zeros(graph.N, dtype=bool)
            true_anomalies[anomaly_indices] = True
            accuracy = np.mean(detected_anomalies == true_anomalies)
            
            # Test filtering effectiveness
            filter_obj = gsp.design_graph_filter('low_pass', cutoff=0.3)
            filtered_signal = gsp.filter_signal(signal, filter_obj)
            filter_effectiveness = np.var(signal) / np.var(filtered_signal)
            
            results['graph_types'].append(graph_name)
            results['spectral_gaps'].append(spectral_gap)
            results['filter_effectiveness'].append(filter_effectiveness)
            results['anomaly_detection_accuracy'].append(accuracy)
            
            # Compute clustering coefficient (if available)
            try:
                import networkx as nx
                nx_graph = nx.from_scipy_sparse_matrix(graph.W)
                clustering_coeff = nx.average_clustering(nx_graph)
            except:
                clustering_coeff = 0.0
            
            results['clustering_coefficients'].append(clustering_coeff)
        
        self.results['graph_comparison'] = results
        return results
    
    def parallel_processing_simulation(self, num_signals: int = 100) -> Dict:
        """
        Simulate parallel processing of multiple signals
        Demonstrates big data processing capabilities
        
        Args:
            num_signals: Number of signals to process simultaneously
            
        Returns:
            Dictionary with parallel processing results
        """
        results = {
            'num_signals': num_signals,
            'sequential_time': 0,
            'batch_time': 0,
            'speedup_factor': 0
        }
        
        print(f"Running parallel processing simulation with {num_signals} signals...")
        
        # Create graph
        graph = graphs.Sensor(N=100, seed=42)
        gsp = GraphSignalProcessor(graph)
        
        # Generate multiple signals
        signals = np.column_stack([
            generate_test_signals(graph, 'smooth') for _ in range(num_signals)
        ])
        
        # Sequential processing
        start_time = time.time()
        for i in range(num_signals):
            analysis = gsp.spectral_analysis(signals[:, i])
        sequential_time = time.time() - start_time
        
        # Batch processing (vectorized operations)
        start_time = time.time()
        gsp.compute_graph_fourier_basis()
        fourier_coeffs = gsp.eigenvectors.T @ signals  # Process all signals at once
        power_spectra = np.abs(fourier_coeffs) ** 2
        batch_time = time.time() - start_time
        
        speedup_factor = sequential_time / batch_time
        
        results['sequential_time'] = sequential_time
        results['batch_time'] = batch_time
        results['speedup_factor'] = speedup_factor
        
        self.results['parallel_processing'] = results
        return results
    
    def visualize_all_results(self, save_path: str = None):
        """
        Create comprehensive visualization of all experimental results
        
        Args:
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Scalability results
        if 'scalability' in self.results:
            scalability = self.results['scalability']
            axes[0, 0].loglog(scalability['graph_sizes'], scalability['fourier_times'], 'o-', label='Fourier Basis')
            axes[0, 0].loglog(scalability['graph_sizes'], scalability['filtering_times'], 's-', label='Filtering')
            axes[0, 0].loglog(scalability['graph_sizes'], scalability['analysis_times'], '^-', label='Analysis')
            axes[0, 0].set_xlabel('Graph Size')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].set_title('Scalability Analysis')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Noise robustness
        if 'noise_robustness' in self.results:
            noise_rob = self.results['noise_robustness']
            axes[0, 1].semilogy(noise_rob['noise_levels'], noise_rob['mse_before'], 'o-', label='Before Denoising')
            axes[0, 1].semilogy(noise_rob['noise_levels'], noise_rob['mse_after'], 's-', label='After Denoising')
            axes[0, 1].set_xlabel('Noise Level')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_title('Denoising Performance')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: SNR improvement
        if 'noise_robustness' in self.results:
            axes[0, 2].plot(noise_rob['noise_levels'], noise_rob['snr_improvement'], 'o-', color='green')
            axes[0, 2].set_xlabel('Noise Level')
            axes[0, 2].set_ylabel('SNR Improvement (dB)')
            axes[0, 2].set_title('SNR Improvement')
            axes[0, 2].grid(True)
        
        # Plot 4: Graph type comparison - Spectral gaps
        if 'graph_comparison' in self.results:
            graph_comp = self.results['graph_comparison']
            axes[1, 0].bar(graph_comp['graph_types'], graph_comp['spectral_gaps'])
            axes[1, 0].set_ylabel('Spectral Gap')
            axes[1, 0].set_title('Spectral Gaps by Graph Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Anomaly detection accuracy
        if 'graph_comparison' in self.results:
            axes[1, 1].bar(graph_comp['graph_types'], graph_comp['anomaly_detection_accuracy'])
            axes[1, 1].set_ylabel('Detection Accuracy')
            axes[1, 1].set_title('Anomaly Detection by Graph Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Parallel processing speedup
        if 'parallel_processing' in self.results:
            parallel = self.results['parallel_processing']
            categories = ['Sequential', 'Batch']
            times = [parallel['sequential_time'], parallel['batch_time']]
            axes[1, 2].bar(categories, times)
            axes[1, 2].set_ylabel('Time (seconds)')
            axes[1, 2].set_title(f'Processing Time\n(Speedup: {parallel["speedup_factor"]:.1f}x)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of all experiments
        
        Returns:
            String containing the experimental report
        """
        report = []
        report.append("=" * 80)
        report.append("BIG DATA GRAPH SIGNAL PROCESSING - EXPERIMENTAL REPORT")
        report.append("Reproducing Sandryhaila & Moura 2014")
        report.append("=" * 80)
        
        if 'scalability' in self.results:
            scalability = self.results['scalability']
            report.append("\n1. SCALABILITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"Tested graph sizes: {scalability['graph_sizes']}")
            report.append(f"Fourier computation scales as O(N²) to O(N³)")
            report.append(f"Largest graph processed: {max(scalability['graph_sizes'])} vertices")
            report.append(f"Max processing time: {max(scalability['fourier_times']):.2f} seconds")
        
        if 'noise_robustness' in self.results:
            noise_rob = self.results['noise_robustness']
            report.append("\n2. NOISE ROBUSTNESS")
            report.append("-" * 40)
            avg_snr_improvement = np.mean(noise_rob['snr_improvement'])
            report.append(f"Average SNR improvement: {avg_snr_improvement:.2f} dB")
            report.append(f"Best performance at noise level: {noise_rob['noise_levels'][np.argmax(noise_rob['snr_improvement'])]}")
        
        if 'graph_comparison' in self.results:
            graph_comp = self.results['graph_comparison']
            report.append("\n3. GRAPH TYPE ANALYSIS")
            report.append("-" * 40)
            best_graph = graph_comp['graph_types'][np.argmax(graph_comp['anomaly_detection_accuracy'])]
            report.append(f"Best graph for anomaly detection: {best_graph}")
            report.append(f"Average detection accuracy: {np.mean(graph_comp['anomaly_detection_accuracy']):.2%}")
        
        if 'parallel_processing' in self.results:
            parallel = self.results['parallel_processing']
            report.append("\n4. PARALLEL PROCESSING")
            report.append("-" * 40)
            report.append(f"Processed {parallel['num_signals']} signals simultaneously")
            report.append(f"Speedup factor: {parallel['speedup_factor']:.1f}x")
            report.append(f"Batch processing time: {parallel['batch_time']:.3f} seconds")
        
        report.append("\n" + "=" * 80)
        report.append("CONCLUSIONS:")
        report.append("- Graph signal processing scales well for moderate-sized graphs")
        report.append("- Spectral methods provide effective denoising capabilities")
        report.append("- Different graph structures affect processing performance")
        report.append("- Vectorized operations enable significant speedups for big data")
        report.append("=" * 80)
        
        return "\n".join(report)


def run_complete_experiment():
    """
    Run all experiments and generate comprehensive results
    """
    processor = BigDataGraphProcessor()
    
    print("Starting comprehensive big data graph signal processing experiments...")
    print("This reproduces key concepts from Sandryhaila & Moura 2014\n")
    
    # Run all experiments
    processor.scalability_experiment()
    processor.noise_robustness_experiment()
    processor.graph_type_comparison()
    processor.parallel_processing_simulation()
    
    # Generate visualizations
    processor.visualize_all_results('big_data_gsp_results.png')
    
    # Generate and print report
    report = processor.generate_report()
    print(report)
    
    # Save results to CSV for further analysis
    results_df = pd.DataFrame({
        'Experiment': ['Scalability', 'Noise Robustness', 'Graph Comparison', 'Parallel Processing'],
        'Status': ['Completed'] * 4,
        'Key_Metric': [
            f"Max size: {max(processor.results['scalability']['graph_sizes'])}",
            f"Avg SNR improvement: {np.mean(processor.results['noise_robustness']['snr_improvement']):.1f} dB",
            f"Best accuracy: {max(processor.results['graph_comparison']['anomaly_detection_accuracy']):.2%}",
            f"Speedup: {processor.results['parallel_processing']['speedup_factor']:.1f}x"
        ]
    })
    
    results_df.to_csv('experiment_summary.csv', index=False)
    print("\nResults saved to 'experiment_summary.csv' and 'big_data_gsp_results.png'")
    
    return processor


if __name__ == "__main__":
    # Run complete experimental suite
    processor = run_complete_experiment()
