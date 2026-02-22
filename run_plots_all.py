#!/usr/bin/env python
"""
Main plotting script with flexible options.

Usage:
  1. Plot all tasks (default):
     python run_plots_all.py
  
  2. Plot specific tasks:
     Edit the 'selected_tasks' list below and run:
     python run_plots_all.py
"""

from plot_delay import CadencePlotter, run_plots, define_plot_tasks

if __name__ == "__main__":
    plotter = CadencePlotter(base_dir="results_cadence")
    
    # ========================================================================
    # CONFIGURATION: Choose what to plot
    # ========================================================================
    
    # Option 1: Plot EVERYTHING (set to True)
    PLOT_ALL = True
    
    # Option 2: Plot SPECIFIC TASKS (used only if PLOT_ALL = False)
    # Uncomment task names below that you want to plot
    # SELECTED_TASKS = [
    #     "constant_slope_sweeps",
    #     "constant_slope_direct_csv",
    #     "constant_slope_linearity",
    #     # "constant_slope_mc_linearity",
    #     # "constant_slope_transients_4bit",
    #     # "constant_slope_transients_5bit",
    #     # "constant_slope_pvt",
    #     # "constant_slope_digital_sweep_mc",
    #     # "constant_slope_histogram",
    #     # "variable_slope_sweeps",
    #     # "variable_slope_linearity",
    #     # "phase_interpolator_sweeps",
    #     # "phase_interpolator_linearity",
    # ]
    
    # ========================================================================
    # Available tasks (for reference):
    # ========================================================================
    print("\n" + "="*70)
    print("AVAILABLE PLOT TASKS")
    print("="*70)
    tasks = define_plot_tasks()
    for i, (task_name, task_config) in enumerate(tasks.items(), 1):
        print(f"\n{i:2d}. {task_name}")
        print(f"    Description: {task_config['description']}")
        print(f"    Files: {len(task_config.get('files', []))} files")
    
    # ========================================================================
    # Run plots
    # ========================================================================
    print("\n" + "="*70)
    if PLOT_ALL:
        print("RUNNING ALL PLOT TASKS")
    else:
        print(f"RUNNING SELECTED PLOT TASKS ({len(SELECTED_TASKS)} tasks)")
    print("="*70 + "\n")
    
    run_plots(plotter, plot_all=PLOT_ALL, selected_tasks=SELECTED_TASKS if not PLOT_ALL else None)
    
    print("\n" + "="*70)
    print("✓ All requested plots completed!")
    print("="*70)
