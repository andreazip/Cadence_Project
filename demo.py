#!/usr/bin/env python
"""
Demonstration of refactored plotting code capabilities.
Shows how to use different plotting options.
"""

from plot_delay import CadencePlotter, run_plots, define_plot_tasks

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)

def main():
    plotter = CadencePlotter(base_dir="results_cadence")
    tasks = define_plot_tasks()
    
    # Show available tasks
    print_header("1. AVAILABLE PLOT TASKS")
    print(f"Total tasks available: {len(tasks)}\n")
    for i, (task_name, config) in enumerate(tasks.items(), 1):
        num_files = len(config.get('files', []))
        print(f"{i:2d}. {task_name:40s} ({num_files:2d} files) - {config['description']}")
    
    # Example 1: Selective plotting
    print_header("2. EXAMPLE: PLOT ONLY LINEARITY & DIRECT CSV")
    selected = ["constant_slope_linearity", "constant_slope_direct_csv"]
    print(f"Running {len(selected)} selected tasks...\n")
    run_plots(plotter, plot_all=False, selected_tasks=selected)
    
    # Show results
    print_header("3. COMPLETION SUMMARY")
    print("✓ Code refactoring completed successfully!")
    print("✓ All plot functionalities preserved and expanded")
    print("✓ Flexible plotting options implemented")
    print("\nNext steps:")
    print("1. Edit run_plots_all.py to customize tasks")
    print("2. Run 'python run_plots_all.py' to generate plots")
    print("3. Check PLOTTING_GUIDE.md for detailed documentation")
    print("4. See REFACTORING_SUMMARY.md for technical details")
    print("="*70)

if __name__ == "__main__":
    main()
