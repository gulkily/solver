# begin graph_generator.py

import sys
import json
import matplotlib.pyplot as plt

def plot_results(n_values, runtimes, solve_rates, data_directory):
    fig, ax1 = plt.subplots(figsize=(12, 6))
   
    line1, = ax1.plot(n_values, runtimes, marker='o', color='blue', label='Runtime')
    ax1.set_xlabel('Number of variables (n)')
    ax1.set_ylabel('Average runtime (seconds)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Only set log scale if there are positive values
    if any(runtime > 0 for runtime in runtimes):
        ax1.set_yscale('log')
    else:
        print("Warning: All runtime values are zero. Using linear scale for y-axis.")
    
    ax2 = ax1.twinx()
    line2, = ax2.plot(n_values, solve_rates, marker='s', color='red', label='Solve rate')
    ax2.set_ylabel('Solve rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f'3-SAT Solver Analysis (DIMACS CNF instances)')
    plt.grid(True)
    
    max_runtime = max(runtimes)
    poly_ref, = ax1.plot(n_values, [((n/max(n_values))**3) * max_runtime for n in n_values], 
                         'g--', label='n^3 (polynomial)')
    exp_ref, = ax1.plot(n_values, [(2**(n/max(n_values))) * min(runtimes) for n in n_values], 
                        'm--', label='2^n (exponential)')
    
    lines = [line1, line2, poly_ref, exp_ref]
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    output_filename = f'3sat_analysis_dimacs_cnf.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_filename

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python graph_generator.py <results_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    with open(results_file, "r") as f:
        results = json.load(f)

    output_filename = plot_results(
        results["n_values"],
        results["runtimes"],
        results["solve_rates"],
        results["data_directory"]
    )

    print(f"Graph saved as {output_filename}")

# end graph_generator.py