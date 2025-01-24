
from __future__ import division  # Must be at the top
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def read_and_filter_data(csv_path, target_max_iter, target_dim):
    """Read data and filter for specified parameters"""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['Dimensions', 'HMS', 'MaxIter', 'ExecutionTime(s)', 
                   'Cores', 'Seed', 'BestFitness']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter for target parameters
    filtered = df[
        (df['MaxIter'] == target_max_iter) &
        (df['Dimensions'] == target_dim)
    ]
    
    return filtered

def plot_parallel_performance(filtered_df, output_filename):
    """Generate speedup and efficiency plots from filtered data"""
    # Get baseline (sequential)
    sequential = filtered_df[filtered_df['ExecutionType'] == 'Sequential']
    if sequential.empty:
        print "Error: No sequential baseline found for these parameters"
        return
        
    baseline_time = sequential['ExecutionTime(s)'].mean()
    
    # Get MPI results
    mpi_runs = filtered_df[filtered_df['ExecutionType'] == 'MPI']
    if mpi_runs.empty:
        print "Error: No MPI runs found for these parameters"
        return
    
    # Group by core count and average
    grouped = mpi_runs.groupby('Cores').agg({
        'ExecutionTime(s)': 'mean',
        'BestFitness': 'mean'
    }).reset_index()
    
    # Calculate metrics
    grouped['Speedup'] = baseline_time / grouped['ExecutionTime(s)']
    grouped['Efficiency'] = grouped['Speedup'] / grouped['Cores']
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Speedup plot
    plt.subplot(1, 2, 1)
    plt.plot(grouped['Cores'], grouped['Speedup'], 'bo-', label='Measured')
    plt.plot(grouped['Cores'], grouped['Cores'], 'k--', label='Ideal')
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup')
    plt.title('Speedup (vs Sequential Baseline)')
    plt.legend()
    plt.grid(True)
    
    # Efficiency plot
    plt.subplot(1, 2, 2)
    plt.plot(grouped['Cores'], grouped['Efficiency'], 'rs-')
    plt.xlabel('Number of Cores')
    plt.ylabel('Efficiency')
    plt.title('Parallel Efficiency')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_filename, format='svg')
    plt.close()
    print "Saved plot to {0}".format(output_filename)

def main():
    parser = argparse.ArgumentParser(description='Analyze parallel performance')
    parser.add_argument('csv_file', help='Path to results CSV')
    parser.add_argument('--max_iter', type=int, required=True, 
                       help='Target MaxIter value')
    parser.add_argument('--dim', type=int, required=True,
                       help='Target dimensions value')
    parser.add_argument('--output', default='performance_plot',
                       help='Output filename prefix')
    
    args = parser.parse_args()
    
    # Load and filter data
    df = read_and_filter_data(args.csv_file, args.max_iter, args.dim)
    
    if df.empty:
        print "No data found for MaxIter={0}, Dimensions={1}".format(args.max_iter, args.dim)
        return
    
    # Generate plot
    output_filename = "{0}_iter{1}_dim{2}.svg".format(args.output, args.max_iter, args.dim)
    plot_parallel_performance(df, output_filename)

if __name__ == '__main__':
    main()