import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# ========================
# ENHANCED CONFIGURATION
# ========================
INPUT_CSV = "../data/harmony_search_results.csv"
OUTPUT_DIR = "../figures/"

# Parameter combinations to analyze
PARAM_CONFIGS = {
    'fixed_dimension': {
        'fixed_dim': 35,
        'varying_iters': [1e7, 5e7, 1e8]
    },
    'fixed_iteration': {
        'fixed_iter': 1e8,
        'varying_dims': [15, 20, 35]
    }
}

COLORS = plt.cm.viridis(np.linspace(0, 1, 5))  # Color palette for plots

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load and preprocess all data"""
    df = pd.read_csv(csv_path)
    
    # Convert numeric columns
    numeric_cols = ['Dimensions', 'HMS', 'MaxIter', 'ExecutionTime(s)', 
                   'Cores', 'Seed', 'BestFitness']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Filter only successful runs
    return df[df['ExecutionType'].isin(['Sequential', 'MPI'])].copy()

def prepare_comparison_data(df: pd.DataFrame, config_type: str) -> pd.DataFrame:
    """Prepare data for comparative analysis"""
    config = PARAM_CONFIGS[config_type]
    
    if config_type == 'fixed_dimension':
        filtered = df[
            (df['Dimensions'] == config['fixed_dim']) &
            (df['MaxIter'].isin(config['varying_iters']))
        ].copy()
        
    elif config_type == 'fixed_iteration':
        filtered = df[
            (df['MaxIter'] == config['fixed_iter']) &
            (df['Dimensions'].isin(config['varying_dims']))
        ].copy()
        
    return filtered.sort_values('MaxIter')

def plot_comparative_analysis(comparison_df: pd.DataFrame, config_type: str) -> None:
    """Create comparative plots for different parameter combinations"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get sequential baselines
    seq_data = comparison_df[comparison_df['ExecutionType'] == 'Sequential']
    
    # Prepare MPI data
    mpi_data = comparison_df[comparison_df['ExecutionType'] == 'MPI']
    grouped = mpi_data.groupby(['Cores', 'MaxIter', 'Dimensions']).agg({
        'ExecutionTime(s)': 'mean'
    }).reset_index()

    # Get core values for optimal lines
    core_values = sorted(grouped['Cores'].unique())
    max_cores = max(core_values) if core_values else 1

    # Determine plot parameters based on config type
    if config_type == 'fixed_dimension':
        fixed_param = PARAM_CONFIGS[config_type]['fixed_dim']
        varying_params = PARAM_CONFIGS[config_type]['varying_iters']
        param_label = 'Iterations'
        fixed_label = f'Dimension={fixed_param}'
        suffix = f"dim{fixed_param}"
        
    elif config_type == 'fixed_iteration':
        fixed_param = PARAM_CONFIGS[config_type]['fixed_iter']
        varying_params = PARAM_CONFIGS[config_type]['varying_dims']
        param_label = 'Dimensions'
        fixed_label = f'Iterations={fixed_param:.0e}'
        suffix = f"iter{int(fixed_param)}"
        
    else:
        return

    # Create separate figures for speedup and efficiency
    for metric in ['speedup', 'efficiency']:
        plt.figure(figsize=(8, 6))
        
        # Plot parameter lines
        for idx, param_value in enumerate(varying_params):
            color = COLORS[idx]
            label = f"{param_value:.0e}" if config_type == 'fixed_dimension' else f"{param_value}"
            
            # Get baseline
            baseline = seq_data[
                (seq_data[param_label.replace('Iterations', 'MaxIter')] == param_value) &
                (seq_data['ExecutionType'] == 'Sequential')
            ]['ExecutionTime(s)'].mean()
            
            # Get MPI results for this parameter
            if config_type == 'fixed_dimension':
                param_data = grouped[grouped['MaxIter'] == param_value]
            else:
                param_data = grouped[grouped['Dimensions'] == param_value]
            
            # Calculate metrics
            param_data = param_data.copy()
            param_data['Speedup'] = baseline / param_data['ExecutionTime(s)']
            param_data['Efficiency'] = param_data['Speedup'] / param_data['Cores']
            
            # Plot the current metric
            if metric == 'speedup':
                plt.plot(param_data['Cores'], param_data['Speedup'], 
                        color=color, marker='o', linestyle='-', 
                        label=label, linewidth=2)
                plt.ylabel('Speedup Ratio')
                plt.title(f'Speedup Comparison ({fixed_label})')
            else:
                plt.plot(param_data['Cores'], param_data['Efficiency'],
                        color=color, marker='s', linestyle='--',
                        label=label, linewidth=2)
                plt.ylabel('Parallel Efficiency')
                plt.title(f'Efficiency Comparison ({fixed_label})')
                plt.ylim(0, 1.2)

        # Add optimal reference lines
        if metric == 'speedup':
            # Ideal linear speedup line
            plt.plot(core_values, core_values, 'k--', label='Optimal', linewidth=1.5)
        else:
            # Perfect efficiency line at 1.0
            plt.axhline(1, color='k', linestyle='--', label='Optimal', linewidth=1.5)

        plt.xlabel('Number of Cores')
        plt.grid(True, alpha=0.3)
        plt.legend(title=param_label)
        plt.tight_layout()
        
        # Save individual files
        filename = f"{OUTPUT_DIR}/{metric}_{config_type}_{suffix}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
def main():
    df = load_and_preprocess(INPUT_CSV)
    
    # Perform comparative analysis
    for config_type in ['fixed_dimension', 'fixed_iteration']:
        comparison_df = prepare_comparison_data(df, config_type)
        
        if not comparison_df.empty:
            plot_comparative_analysis(comparison_df, config_type)
        else:
            print(f"No data found for {config_type} analysis")

if __name__ == '__main__':
    main()