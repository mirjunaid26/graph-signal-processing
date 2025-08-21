"""
Graph Signal Processing Implementation
Reproducing: "Big Data Analysis with Signal Processing on Graphs" 
by Sandryhaila & Moura (2014)

This module implements the core concepts from the paper:
- Graph Fourier Transform (GFT)
- Graph spectral analysis
- Graph filtering
- Signal processing on irregular graph structures
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
from pygsp import graphs, filters, plotting
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class GraphSignalProcessor:
    """
    Main class implementing graph signal processing methods from Sandryhaila & Moura 2014
    """
    
    def __init__(self, graph: Optional[graphs.Graph] = None):
        """
        Initialize the Graph Signal Processor
        
        Args:
            graph: PyGSP graph object. If None, will be set later.
        """
        self.graph = graph
        self.eigenvalues = None
        self.eigenvectors = None
        self.is_computed = False
        
    def set_graph(self, graph: graphs.Graph):
        """Set the graph for processing"""
        self.graph = graph
        self.is_computed = False
        
    def compute_graph_fourier_basis(self):
        """
        Compute the Graph Fourier Transform basis (eigendecomposition of Laplacian)
        This is fundamental to the Sandryhaila & Moura approach
        """
        if self.graph is None:
            raise ValueError("Graph must be set before computing Fourier basis")
            
        # Compute Laplacian eigendecomposition
        self.graph.compute_fourier_basis()
        self.eigenvalues = self.graph.e  # Eigenvalues (frequencies)
        self.eigenvectors = self.graph.U  # Eigenvectors (Fourier modes)
        self.is_computed = True
        
        return self.eigenvalues, self.eigenvectors
    
    def graph_fourier_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Graph Fourier Transform of a signal
        
        Args:
            signal: Signal defined on graph vertices (N,) or (N, K) for K signals
            
        Returns:
            Fourier coefficients in spectral domain
        """
        if not self.is_computed:
            self.compute_graph_fourier_basis()
            
        # GFT: \hat{f} = U^T * f
        return self.eigenvectors.T @ signal
    
    def inverse_graph_fourier_transform(self, fourier_coeffs: np.ndarray) -> np.ndarray:
        """
        Compute Inverse Graph Fourier Transform
        
        Args:
            fourier_coeffs: Fourier coefficients in spectral domain
            
        Returns:
            Signal in vertex domain
        """
        if not self.is_computed:
            raise ValueError("Fourier basis must be computed first")
            
        # IGFT: f = U * \hat{f}
        return self.eigenvectors @ fourier_coeffs
    
    def design_graph_filter(self, filter_type: str = 'low_pass', 
                           cutoff: float = 0.3, order: int = 10) -> filters.Filter:
        """
        Design graph filters as described in the paper
        
        Args:
            filter_type: 'low_pass', 'high_pass', 'band_pass'
            cutoff: Cutoff frequency (normalized)
            order: Filter order
            
        Returns:
            PyGSP filter object
        """
        if not self.is_computed:
            self.compute_graph_fourier_basis()
            
        if filter_type == 'low_pass':
            g = filters.Heat(self.graph, tau=cutoff)
        elif filter_type == 'high_pass':
            g = filters.Heat(self.graph, tau=cutoff)
            # Convert to high-pass by subtracting from identity
            def high_pass_response(x):
                return 1 - g.evaluate(x)
            g = filters.Filter(self.graph, high_pass_response)
        elif filter_type == 'band_pass':
            g = filters.MexicanHat(self.graph, Nf=order)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        return g
    
    def filter_signal(self, signal: np.ndarray, graph_filter: filters.Filter) -> np.ndarray:
        """
        Apply graph filter to signal
        
        Args:
            signal: Input signal on graph vertices
            graph_filter: Graph filter to apply
            
        Returns:
            Filtered signal
        """
        return graph_filter.filter(signal)
    
    def spectral_analysis(self, signal: np.ndarray) -> dict:
        """
        Perform spectral analysis of graph signal as described in the paper
        
        Args:
            signal: Signal defined on graph vertices
            
        Returns:
            Dictionary with spectral analysis results
        """
        if not self.is_computed:
            self.compute_graph_fourier_basis()
            
        # Compute GFT
        fourier_coeffs = self.graph_fourier_transform(signal)
        
        # Compute power spectral density
        power_spectrum = np.abs(fourier_coeffs) ** 2
        
        # Compute energy in different frequency bands
        n_freqs = len(self.eigenvalues)
        low_freq_energy = np.sum(power_spectrum[:n_freqs//3])
        mid_freq_energy = np.sum(power_spectrum[n_freqs//3:2*n_freqs//3])
        high_freq_energy = np.sum(power_spectrum[2*n_freqs//3:])
        
        total_energy = np.sum(power_spectrum)
        
        return {
            'fourier_coefficients': fourier_coeffs,
            'power_spectrum': power_spectrum,
            'eigenvalues': self.eigenvalues,
            'total_energy': total_energy,
            'low_freq_energy_ratio': low_freq_energy / total_energy,
            'mid_freq_energy_ratio': mid_freq_energy / total_energy,
            'high_freq_energy_ratio': high_freq_energy / total_energy
        }
    
    def detect_graph_anomalies(self, signal: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """
        Detect anomalies in graph signals using spectral methods
        Implementation of anomaly detection concepts from the paper
        
        Args:
            signal: Input signal
            threshold: Anomaly detection threshold (in standard deviations)
            
        Returns:
            Binary array indicating anomalous vertices
        """
        # Apply low-pass filter to get smooth baseline
        low_pass_filter = self.design_graph_filter('low_pass', cutoff=0.2)
        smooth_signal = self.filter_signal(signal, low_pass_filter)
        
        # Compute residual (high-frequency components)
        residual = signal - smooth_signal
        
        # Detect anomalies based on residual magnitude
        residual_std = np.std(residual)
        anomalies = np.abs(residual) > threshold * residual_std
        
        return anomalies
    
    def graph_signal_denoising(self, noisy_signal: np.ndarray, 
                              noise_level: float = 0.1) -> np.ndarray:
        """
        Denoise graph signals using spectral filtering
        
        Args:
            noisy_signal: Noisy input signal
            noise_level: Estimated noise level for filter design
            
        Returns:
            Denoised signal
        """
        # Design adaptive low-pass filter based on noise level
        cutoff = max(0.1, 1.0 - noise_level)
        denoising_filter = self.design_graph_filter('low_pass', cutoff=cutoff)
        
        # Apply filter
        denoised_signal = self.filter_signal(noisy_signal, denoising_filter)
        
        return denoised_signal


def create_example_graphs() -> dict:
    """
    Create example graphs for testing the implementation
    Following examples similar to those in the paper
    """
    graphs_dict = {}
    
    # 1. Sensor Network Graph
    graphs_dict['sensor_network'] = graphs.Sensor(N=100, seed=42)
    
    # 2. Community Graph (social network-like)
    graphs_dict['community'] = graphs.Community(N=100, seed=42)
    
    # 3. Grid Graph (regular structure)
    graphs_dict['grid'] = graphs.Grid2d(N1=10, N2=10)
    
    # 4. Random Graph
    graphs_dict['random'] = graphs.ErdosRenyi(N=100, p=0.1, seed=42)
    
    # 5. Small World Graph
    graphs_dict['small_world'] = graphs.BarabasiAlbert(N=100, seed=42)
    
    return graphs_dict


def generate_test_signals(graph: graphs.Graph, signal_type: str = 'smooth') -> np.ndarray:
    """
    Generate test signals on graphs for validation
    
    Args:
        graph: PyGSP graph object
        signal_type: Type of signal to generate
        
    Returns:
        Generated signal
    """
    N = graph.N
    np.random.seed(42)
    
    if signal_type == 'smooth':
        # Generate smooth signal (low-frequency)
        graph.compute_fourier_basis()
        coeffs = np.zeros(N)
        coeffs[:N//4] = np.random.randn(N//4)  # Only low frequencies
        signal = graph.U @ coeffs
        
    elif signal_type == 'noisy':
        # Generate smooth signal + noise
        smooth_signal = generate_test_signals(graph, 'smooth')
        noise = 0.3 * np.random.randn(N)
        signal = smooth_signal + noise
        
    elif signal_type == 'piecewise':
        # Generate piecewise constant signal
        signal = np.random.choice([0, 1, 2], size=N)
        
    elif signal_type == 'random':
        # Random signal
        signal = np.random.randn(N)
        
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return signal


def visualize_results(gsp: GraphSignalProcessor, signal: np.ndarray, 
                     analysis_results: dict, save_path: str = None):
    """
    Visualize graph signal processing results
    
    Args:
        gsp: GraphSignalProcessor instance
        signal: Original signal
        analysis_results: Results from spectral analysis
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original signal on graph
    axes[0, 0].scatter(range(len(signal)), signal, alpha=0.7)
    axes[0, 0].set_title('Original Signal on Graph')
    axes[0, 0].set_xlabel('Vertex Index')
    axes[0, 0].set_ylabel('Signal Value')
    
    # Plot 2: Power spectrum
    axes[0, 1].plot(gsp.eigenvalues, analysis_results['power_spectrum'])
    axes[0, 1].set_title('Graph Power Spectrum')
    axes[0, 1].set_xlabel('Graph Frequency (Eigenvalue)')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Fourier coefficients
    axes[1, 0].stem(range(len(analysis_results['fourier_coefficients'])), 
                    np.abs(analysis_results['fourier_coefficients']))
    axes[1, 0].set_title('Graph Fourier Coefficients')
    axes[1, 0].set_xlabel('Frequency Index')
    axes[1, 0].set_ylabel('|Coefficient|')
    
    # Plot 4: Energy distribution
    energy_ratios = [
        analysis_results['low_freq_energy_ratio'],
        analysis_results['mid_freq_energy_ratio'],
        analysis_results['high_freq_energy_ratio']
    ]
    axes[1, 1].bar(['Low', 'Mid', 'High'], energy_ratios)
    axes[1, 1].set_title('Energy Distribution by Frequency Band')
    axes[1, 1].set_ylabel('Energy Ratio')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage demonstrating the paper's concepts
    print("Graph Signal Processing - Sandryhaila & Moura 2014 Reproduction")
    print("=" * 60)
    
    # Create example graphs
    graphs_dict = create_example_graphs()
    
    # Test with sensor network graph
    graph = graphs_dict['sensor_network']
    gsp = GraphSignalProcessor(graph)
    
    # Generate test signal
    signal = generate_test_signals(graph, 'smooth')
    
    # Perform spectral analysis
    analysis = gsp.spectral_analysis(signal)
    
    print(f"Graph: {graph.N} vertices, {graph.Ne} edges")
    print(f"Total signal energy: {analysis['total_energy']:.2f}")
    print(f"Low frequency energy: {analysis['low_freq_energy_ratio']:.2%}")
    print(f"Mid frequency energy: {analysis['mid_freq_energy_ratio']:.2%}")
    print(f"High frequency energy: {analysis['high_freq_energy_ratio']:.2%}")
