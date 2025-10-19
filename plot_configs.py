#!/usr/bin/env python3
"""
Plot per-GPU rates for different DistServe configurations.
"""
# python plot_configs.py sim_result.txt

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_sim_results(filename):
    """Parse the simulation results file."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by Namespace blocks
    blocks = content.split('Namespace(')
    
    all_configs = []
    
    for block in blocks[1:]:  # Skip first empty part
        # Extract metadata from Namespace
        lines = block.split('\n')
        
        # Find the data lines (TSV format)
        for i, line in enumerate(lines):
            if line.startswith('pp_cross\t'):
                # Found header, now read data lines
                for j in range(i + 1, len(lines)):
                    data_line = lines[j].strip()
                    if not data_line or data_line.startswith('Namespace'):
                        break
                    
                    # Parse the TSV line
                    parts = data_line.split('\t')
                    if len(parts) >= 7:
                        try:
                            pp_cross = int(parts[0])
                            tp_prefill = int(parts[1])
                            pp_prefill = int(parts[2])
                            tp_decode = int(parts[3])
                            pp_decode = int(parts[4])
                            total_gpus = int(parts[5])
                            per_gpu_rate = float(parts[6])
                            
                            config = {
                                'pp_cross': pp_cross,
                                'tp_prefill': tp_prefill,
                                'pp_prefill': pp_prefill,
                                'tp_decode': tp_decode,
                                'pp_decode': pp_decode,
                                'total_gpus': total_gpus,
                                'per_gpu_rate': per_gpu_rate,
                                'config_tuple': (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode, total_gpus),
                                'config_str': f"({pp_cross},{tp_prefill},{pp_prefill},{tp_decode},{pp_decode},{total_gpus})"
                            }
                            all_configs.append(config)
                        except (ValueError, IndexError):
                            continue
                break
    
    return all_configs


def plot_configs(configs, filename='config_performance.png'):
    """Create scatter plot of configurations vs per-GPU rate."""
    
    # Filter out configs with 0 per_gpu_rate (failed configs)
    valid_configs = [c for c in configs if c['per_gpu_rate'] > 0]
    
    if not valid_configs:
        print("No valid configurations found!")
        return
    
    # Sort by total_gpus for better visualization
    valid_configs.sort(key=lambda x: (x['total_gpus'], x['per_gpu_rate']))
    
    # Prepare data for plotting
    config_labels = [c['config_str'] for c in valid_configs]
    per_gpu_rates = [c['per_gpu_rate'] for c in valid_configs]
    total_gpus = [c['total_gpus'] for c in valid_configs]
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Create scatter plot with color based on total_gpus
    scatter = plt.scatter(range(len(config_labels)), per_gpu_rates, 
                         c=total_gpus, cmap='viridis', s=100, alpha=0.6, 
                         edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total GPUs', fontsize=12)
    
    # Set labels and title
    plt.xlabel('Configuration (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode, total_gpus)', fontsize=12)
    plt.ylabel('Per-GPU Rate (req/s/GPU)', fontsize=12)
    plt.title('DistServe Configuration Performance', fontsize=16, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(config_labels)), config_labels, rotation=90, fontsize=8)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal line for mean
    mean_rate = np.mean(per_gpu_rates)
    plt.axhline(y=mean_rate, color='r', linestyle='--', alpha=0.5, 
                label=f'Mean: {mean_rate:.2f} req/s/GPU')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total configurations: {len(valid_configs)}")
    print(f"Best per-GPU rate: {max(per_gpu_rates):.2f} req/s/GPU")
    print(f"Config: {valid_configs[per_gpu_rates.index(max(per_gpu_rates))]['config_str']}")
    print(f"Worst per-GPU rate: {min(per_gpu_rates):.2f} req/s/GPU")
    print(f"Mean per-GPU rate: {mean_rate:.2f} req/s/GPU")


if __name__ == '__main__':
    import sys
    
    filename = sys.argv[1] if len(sys.argv) > 1 else 'sim_result.txt'
    
    print(f"Reading {filename}...")
    configs = parse_sim_results(filename)
    print(f"Found {len(configs)} configurations")
    
    plot_configs(configs)
    
    plt.show()