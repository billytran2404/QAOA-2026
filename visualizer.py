import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime

# Configuration
cost_history_plot = True
parameter_history_plot = True
heatmap_plot = True
evolution_plot = True

def load_results(json_path: str) -> dict:
    """Load results from JSON file."""
    print(f"Loading results from: {json_path}")
    with open(json_path, "r") as f:
        stats = json.load(f)
    return stats

def convert_ortools_to_qaoa_format(ortools_bitstring: List[int]) -> List[int]:
    """Convert OR-Tools binary solution to QAOA format (already in 0/1 format)."""
    return list(ortools_bitstring)

def top_k_lists(list_of_lists: List[List], k: int = 20) -> List[Tuple[Tuple, int, float]]:
    """Get top k most common lists from QAOA samples."""
    total = len(list_of_lists)
    counts = Counter(map(tuple, list_of_lists))
    top_k = counts.most_common(k)
    return [(lst, cnt, cnt / total) for lst, cnt in top_k]

def plot_top_k_with_ortools(top_k_stats, ortools_bitstring, output_dir, timestamp, qbit_match):
    """Plot top K QAOA results with OR-Tools result highlighted or separate."""
    plt.rcParams['pdf.fonttype'] = 42
    
    # Convert bitstrings for display
    qaoa_labels = [list(tup) for tup, _, _ in top_k_stats]
    qaoa_counts = [cnt for _, cnt, _ in top_k_stats]
    qaoa_rates = [rate for _, _, rate in top_k_stats]
    
    ortools_label = convert_ortools_to_qaoa_format(ortools_bitstring)
    ortools_bitstring_tuple = tuple(ortools_label)
    
    # Check if OR-Tools result is in top K (compare with converted labels)
    qaoa_label_tuples = [tuple(lbl) for lbl in qaoa_labels]
    ortools_in_top_k = ortools_bitstring_tuple in qaoa_label_tuples
    
    # Find the index of OR-Tools result in top K if it exists
    ortools_index = None
    if ortools_in_top_k:
        for idx, lbl_tuple in enumerate(qaoa_label_tuples):
            if lbl_tuple == ortools_bitstring_tuple:
                ortools_index = idx
                break
    
    # Build display data
    if ortools_in_top_k:
        # OR-Tools is in top K, highlight it in orange and mark it
        labels = qaoa_labels
        counts = qaoa_counts
        rates = qaoa_rates
        colors = ['orange' if tuple(lbl) == ortools_bitstring_tuple else 'C0' for lbl in labels]
        title_suffix = " (OR-Tools in Top K)"
    else:
        # OR-Tools is not in top K, add as extra bar
        labels = qaoa_labels + [ortools_label]
        counts = qaoa_counts + [1]  # OR-Tools gets count of 1 (single solution)
        rates = qaoa_rates + [1 / 20000]  # Single solution out of 20000 shots
        colors = ['C0'] * len(qaoa_labels) + ['orange']
        title_suffix = " (OR-Tools as Extra Entry)"
        ortools_index = len(qaoa_labels)
    
    # Create plot
    plt.figure(figsize=(10, max(6, len(labels) / 3)))
    bars = plt.barh(range(len(labels)), rates, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Appearance rate (or relative frequency)")
    # plt.title(f"Top QAOA Sampling Results {title_suffix}\n(Orange = OR-Tools Result)")
    
    # Annotate bars with count and rate
    for i, (b, cnt, rate) in enumerate(zip(bars, counts, rates)):
        # Check if this is the OR-Tools result
        if i == ortools_index:
            label_text = f"OR-Tools: {cnt} ({rate:.2%})"
        else:
            label_text = f"{cnt} ({rate:.2%})"
        plt.text(b.get_width(), b.get_y() + b.get_height()/2,
                 label_text, ha="left", va="center", fontsize=8)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"plot_merged_{qbit_match}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.margins(x=0.2)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Top K plot saved to: {plot_path}")
    plt.close()
    
    return ortools_in_top_k

def plot_cost_history(optimization_history, output_dir, timestamp, qbit_match):
    """Plot QAOA cost function over iterations."""
    plt.rcParams['pdf.fonttype'] = 42
    
    iterations = [entry["iteration"] for entry in optimization_history]
    costs = [entry["f_x"] for entry in optimization_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, costs, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    # plt.title("QAOA Cost Function Value over Iterations")
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"cost_history_{qbit_match}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Cost history plot saved to: {plot_path}")
    plt.close()

def plot_parameter_history(optimization_history, output_dir, timestamp, qbit_match):
    """Plot QAOA parameters over iterations."""
    plt.rcParams['pdf.fonttype'] = 42
    
    gammas = [x[0] for x in [entry["x"] for entry in optimization_history]]
    betas = [x[1] for x in [entry["x"] for entry in optimization_history]]
    
    iterations = list(range(1, len(gammas) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, gammas, marker='o', label='Gamma')
    plt.plot(iterations, betas, marker='s', label='Beta')
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    # plt.title("QAOA Parameters over Iterations")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"parameter_history_{qbit_match}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Parameter history plot saved to: {plot_path}")
    plt.close()

def plot_cost_landscape(landscape_data, output_dir, timestamp, qbit_match):
    """Plot QAOA cost landscape in 2D and 3D from precomputed data."""
    plt.rcParams['pdf.fonttype'] = 42

    gamma_range = np.array(landscape_data["gamma_range"])
    beta_range = np.array(landscape_data["beta_range"])
    cost_landscape = np.array(landscape_data["values"])

    os.makedirs(output_dir, exist_ok=True)

    # 2D heatmap
    fig_2d = plt.figure(figsize=(10, 8))
    plt.imshow(
        cost_landscape,
        extent=[beta_range.min(), beta_range.max(), gamma_range.min(), gamma_range.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Cost Function (f(x))')
    plt.xlabel('Beta Parameter')
    plt.ylabel('Gamma Parameter')
    # plt.title('QAOA Cost Function Heatmap (Gamma vs Beta - 2D)')
    plot_filename_2d = f"cost_landscape_2d_{qbit_match}.pdf"
    plot_path_2d = os.path.join(output_dir, plot_filename_2d)
    plt.savefig(plot_path_2d, format='pdf', bbox_inches='tight')
    print(f"2D cost landscape plot saved to: {plot_path_2d}")
    plt.close()

    # 3D surface plot
    fig_3d = plt.figure(figsize=(12, 9))
    ax = fig_3d.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(beta_range, gamma_range)
    surf = ax.plot_surface(X, Y, cost_landscape, cmap='viridis', alpha=0.8, edgecolor='none')
    fig_3d.colorbar(surf, ax=ax, label='Cost Function (f(x))')
    ax.set_xlabel('Beta Parameter')
    ax.set_ylabel('Gamma Parameter')
    ax.set_zlabel('Cost Function Value')
    # ax.set_title('QAOA Cost Function Landscape (Gamma vs Beta - 3D)')
    plot_filename_3d = f"cost_landscape_3d_{qbit_match}.pdf"
    plot_path_3d = os.path.join(output_dir, plot_filename_3d)
    plt.savefig(plot_path_3d, format='pdf', bbox_inches='tight')
    print(f"3D cost landscape plot saved to: {plot_path_3d}")
    plt.close()


def align_symmetry_branch(gammas, betas, period_gamma=2 * np.pi, period_beta=np.pi):
    """Adjusts QAOA parameters to remain in a continuous principal symmetry sector."""
    g = gammas % period_gamma
    b = betas % period_beta
    if np.mean(g) > (period_gamma / 2):
        g = period_gamma - g
        b = period_beta - b
    g = np.unwrap(g)
    b = np.unwrap(b * 2) / 2
    return g, b


def plot_evolution(evolution_data, output_dir, timestamp, qbit_match):
    """Plot Vikstal/Zhou parameter evolution across depths."""
    import matplotlib.colors as mcolors
    from collections import defaultdict
    plt.rcParams['pdf.fonttype'] = 42

    records = evolution_data.get("records", [])
    if not records:
        print("No evolution records found — skipping evolution plot.")
        return

    # Group records by depth
    by_depth = defaultdict(list)
    for rec in records:
        by_depth[rec["depth"]].append(rec)
    for d in by_depth:
        by_depth[d].sort(key=lambda r: r["layer"])

    available_depths = sorted(by_depth.keys())
    # Prefer depths 3, 4, 5; fall back to whatever is available
    target_depths = [d for d in [3, 4, 5] if d in available_depths] or available_depths

    n_cols = len(target_depths)
    fig, axes = plt.subplots(nrows=2, ncols=n_cols,
                             figsize=(max(10, 3 * n_cols), 6),
                             sharex=False, sharey="row")
    # Normalise axes to always be 2D array
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    color_gamma = '#1f77b4'
    color_beta = '#ff7f0e'
    face_color_gamma = mcolors.to_rgba(color_gamma, alpha=0.4)
    face_color_beta = mcolors.to_rgba(color_beta, alpha=0.4)

    for col_idx, depth in enumerate(target_depths):
        group = by_depth[depth]
        raw_gammas = np.array([r["gamma_raw"] for r in group])
        raw_betas = np.array([r["beta_raw"] for r in group])
        layers = np.array([r["layer"] for r in group])

        g_aligned, b_aligned = align_symmetry_branch(raw_gammas, raw_betas)
        g_norm = g_aligned / np.pi
        b_norm = b_aligned / np.pi

        ax_gamma = axes[0, col_idx]
        ax_beta = axes[1, col_idx]

        ax_gamma.plot(layers, g_norm, marker="o", linestyle="-", color=color_gamma,
                      markersize=6, markerfacecolor=face_color_gamma,
                      markeredgecolor=color_gamma, markeredgewidth=1.5)
        ax_gamma.set_title(f"Depth $p={depth}$")
        ax_gamma.set_xticks(layers)
        ax_gamma.grid(True, linestyle="--", alpha=0.5)

        ax_beta.plot(layers, b_norm, marker="s", linestyle="-", color=color_beta,
                     markersize=6, markerfacecolor=face_color_beta,
                     markeredgecolor=color_beta, markeredgewidth=1.5)
        ax_beta.set_xticks(layers)
        ax_beta.grid(True, linestyle="--", alpha=0.5)

        if col_idx == 0:
            ax_gamma.set_ylabel(r"$\gamma / \pi$", fontsize=12)
            ax_beta.set_ylabel(r"$\beta / \pi$", fontsize=12)
        ax_beta.set_xlabel("Layer $i$", fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"evolution_plot_{qbit_match}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Evolution plot saved to: {plot_path}")
    plt.close(fig)


def generate_visualizations(json_path: str):
    """Generate all visualizations from saved results."""
    
    # Load results
    stats = load_results(json_path)
    
    # Extract data
    problem_info = stats["problem_info"]
    qaoa_data = stats["qaoa"]
    ortools_data = stats["ortools"]
    timestamp = stats["timestamp"]
    
    file_path = problem_info["file_path"]
    n_qubits = problem_info["n_qubits"]
    qaoa_samples = qaoa_data["samples"]
    ortools_solution = [int(x) for x in ortools_data["solution"]]
    optimization_history = qaoa_data["optimization_history"]
    
    # Extract filename for naming plots
    qbit_match = file_path.split('_')[-1].replace('.xlsx', '')
    
    # Create output directories
    top_k_dir = os.path.join(os.path.dirname(__file__), "output", "top_k")
    cost_history_dir = os.path.join(os.path.dirname(__file__), "output", "cost_history")
    parameter_history_dir = os.path.join(os.path.dirname(__file__), "output", "parameter_history")
    cost_landscape_dir = os.path.join(os.path.dirname(__file__), "output", "cost_landscape")
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Generate top K plot
    print("\nGenerating top K plot with OR-Tools integration...")
    cnt = top_k_lists(list_of_lists=qaoa_samples, k=20)
    ortools_in_top_k = plot_top_k_with_ortools(
        cnt, ortools_solution, top_k_dir, timestamp, qbit_match
    )
    
    # Generate cost history plot
    if cost_history_plot and optimization_history:
        print("Generating cost history plot...")
        plot_cost_history(optimization_history, cost_history_dir, timestamp, qbit_match)
    
    # Generate parameter history plot
    if parameter_history_plot and optimization_history:
        print("Generating parameter history plot...")
        plot_parameter_history(optimization_history, parameter_history_dir, timestamp, qbit_match)
    
    # Generate cost landscape plots
    if heatmap_plot:
        landscape_data = qaoa_data.get("cost_landscape")
        if landscape_data:
            print("Generating cost landscape plots...")
            plot_cost_landscape(landscape_data, cost_landscape_dir, timestamp, qbit_match)
        else:
            print("Note: No cost landscape data found in results. Re-run algorithm_runner.py to generate it.")

    # Generate evolution plot
    if evolution_plot:
        evolution_data_json = stats.get("evolution")
        if evolution_data_json and evolution_data_json.get("records"):
            print("Generating evolution plot...")
            evolution_dir = os.path.join(os.path.dirname(__file__), "output", "vikstal_evolution")
            plot_evolution(evolution_data_json, evolution_dir, timestamp, qbit_match)
        else:
            print("Note: No evolution data found. Set compute_evolution=True in algorithm_runner.py to generate it.")

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"QAOA Optimal Cost: {qaoa_data['optimal_cost']:.6f}")
    print(f"OR-Tools Optimal Cost: {ortools_data['optimal_cost']:.6f}")
    print(f"QAOA Execution Time: {qaoa_data['execution_time']:.2f} seconds")
    print(f"OR-Tools Execution Time: {ortools_data['execution_time']:.2f} seconds")
    print(f"OR-Tools in Top 20 QAOA Samples: {ortools_in_top_k}")
    print(f"OR-Tools Solution: {ortools_solution}")
    print(f"QAOA Top Solution: {list(cnt[0][0])}")
    print("="*60)
    print(f"\nAll visualizations saved to output/ directory")

def find_latest_results() -> str:
    """Find the latest algorithm results JSON file."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    json_files = sorted(Path(output_dir).glob("algorithm_results_*.json"), 
                       key=os.path.getmtime, reverse=True)
    
    if not json_files:
        raise FileNotFoundError(f"No algorithm_results_*.json files found in {output_dir}")
    
    return str(json_files[0])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use provided JSON file path
        json_path = sys.argv[1]
    else:
        # Find and use latest results
        json_path = find_latest_results()
        print(f"Using latest results: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    generate_visualizations(json_path)
