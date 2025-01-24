import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Optional, Tuple

def read_and_filter_data(csv_path: str, target_max_iter: int, target_dim: int) -> pd.DataFrame:
    """Read data and filter for specified parameters"""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['Dimensions', 'HMS', 'MaxIter', 'ExecutionTime(s)', 
                   'Cores', 'Seed', 'BestFitness']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    return df[
        (df['MaxIter'] == target_max_iter) &
        (df['Dimensions'] == target_dim)
    ]

def plot_parallel_performance(filtered_df: pd.DataFrame, output_filename: str) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Generate speedup and efficiency plots from filtered data"""
    # Get baseline (sequential)
    sequential = filtered_df[filtered_df['ExecutionType'] == 'Sequential']
    if sequential.empty:
        print(f"Error: No sequential baseline found")
        return None
        
    baseline_time = sequential['ExecutionTime(s)'].mean()
    
    # Get MPI results
    mpi_runs = filtered_df[filtered_df['ExecutionType'] == 'MPI']
    if mpi_runs.empty:
        print(f"Error: No MPI runs found")
        return None
    
    # Group by core count and average
    grouped = mpi_runs.groupby('Cores').agg({
        'ExecutionTime(s)': 'mean',
        'BestFitness': 'mean'
    }).reset_index()
    
    # Calculate metrics
    grouped['Speedup'] = baseline_time / grouped['ExecutionTime(s)']
    grouped['Efficiency'] = grouped['Speedup'] / grouped['Cores']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Speedup plot
    ax1.plot(grouped['Cores'], grouped['Speedup'], 'bo-', label='Measured')
    ax1.plot(grouped['Cores'], grouped['Cores'], 'k--', label='Ideal')
    ax1.set(xlabel='Number of Cores', ylabel='Speedup', title='Speedup (vs Sequential Baseline)')
    ax1.legend()
    ax1.grid(True)
    
    # Efficiency plot
    ax2.plot(grouped['Cores'], grouped['Efficiency'], 'rs-')
    ax2.set(xlabel='Number of Cores', ylabel='Efficiency', title='Parallel Efficiency')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved plot to {output_filename}")
    return fig, (ax1, ax2)

def main():
    parser = argparse.ArgumentParser(description='Analyze parallel performance')
    parser.add_argument('csv_file', help='harmony_search_results.csv')
    parser.add_argument('--max_iter', type=int, required=True, 
                       help='Target MaxIter value')
    parser.add_argument('--dim', type=int, required=True,
                       help='Target dimensions value')
    parser.add_argument('--output', default='performance_plot',
                       help='Output filename prefix')
    
    args = parser.parse_args()
    
    df = read_and_filter_data(args.csv_file, args.max_iter, args.dim)
    
    if df.empty:
        print(f"⚠️ No data for MaxIter={args.max_iter}, Dimensions={args.dim}")
        return
    
    output_filename = f"../figures/{args.output}_iter{args.max_iter}_dim{args.dim}.svg"
    plot_parallel_performance(df, output_filename)

if __name__ == '__main__':
    main()