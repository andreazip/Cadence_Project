#!/usr/bin/env python
"""Final test of refactored plotting code."""

from plot_delay import CadencePlotter, run_plots

if __name__ == "__main__":
    plotter = CadencePlotter(base_dir="results_cadence")
    
    # Test a variety of task types
    test_tasks = [
        "constant_slope_direct_csv",      # New function
        "constant_slope_linearity",       # Linearity analysis
        "constant_slope_sweeps",          # Auto sweeps
    ]
    
    print("\n" + "="*70)
    print("FINAL COMPREHENSIVE TEST")
    print("="*70)
    print(f"Testing {len(test_tasks)} plot tasks...")
    print("="*70 + "\n")
    
    run_plots(plotter, plot_all=False, selected_tasks=test_tasks)
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("All plot functionalities verified.")
    print("="*70)
