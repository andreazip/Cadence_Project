#!/usr/bin/env python
"""Test selective plotting functionality."""

from plot_delay import CadencePlotter, run_plots

if __name__ == "__main__":
    plotter = CadencePlotter(base_dir="results_cadence")
    
    # Test: Plot only specific tasks
    selected = [
        "constant_slope_direct_csv",
        "constant_slope_linearity"
    ]
    
    print("\n" + "="*70)
    print("Testing SELECTIVE PLOT mode (2 tasks)")
    print("="*70 + "\n")
    
    run_plots(plotter, plot_all=False, selected_tasks=selected)
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)
