# Plot Cadence - Refactored Code Guide

## Overview
The plotting code has been refactored for better organization, maintainability, and flexibility. All original functionalities are preserved.

## Key Improvements

✅ **Organized Plot Tasks** - All plotting tasks are organized in a structured configuration
✅ **Selective Plotting** - Choose to plot all tasks or specific ones
✅ **Better Error Handling** - Individual task errors won't stop the entire run
✅ **Clear Task Descriptions** - Each task has a description and file list
✅ **Cleaner Code** - Removed redundancy and empty sections

## Available Plot Tasks

### Constant Slope (CS) Tasks
1. **constant_slope_sweeps** - Automatic sweeps for all CS files
2. **constant_slope_direct_csv** - Direct CSV sweeps with decimal codes
3. **constant_slope_linearity** - DNL/INL linearity analysis
4. **constant_slope_mc_linearity** - MC linearity per iteration
5. **constant_slope_transients_4bit** - 4-bit transient signals
6. **constant_slope_transients_5bit** - 5-bit transient signals
7. **constant_slope_pvt** - PVT analysis (Temperature, VDD)
8. **constant_slope_digital_sweep_mc** - Digital sweep MC all iterations
9. **constant_slope_histogram** - DR/Resolution histogram

### Variable Slope (DLCSI) Tasks
10. **variable_slope_sweeps** - Automatic sweeps
11. **variable_slope_linearity** - Linearity analysis

### Phase Interpolator (PI) Tasks
12. **phase_interpolator_sweeps** - Automatic sweeps
13. **phase_interpolator_linearity** - Linearity analysis

## Usage

### Method 1: Plot Everything
```bash
cd c:\Users\zipar\MEP_cadence
.\plot_cadence\Scripts\Activate.ps1
python run_plots_all.py
```

### Method 2: Plot Specific Tasks
Edit `run_plots_all.py` and modify the `SELECTED_TASKS` list:

```python
PLOT_ALL = False

SELECTED_TASKS = [
    "constant_slope_sweeps",
    "constant_slope_linearity",
    "constant_slope_direct_csv"
]
```

Then run:
```bash
python run_plots_all.py
```

### Method 3: Direct Python Usage
```python
from plot_delay import CadencePlotter, run_plots

plotter = CadencePlotter(base_dir="results_cadence")

# Plot all
run_plots(plotter, plot_all=True)

# Plot specific tasks
run_plots(plotter, plot_all=False, selected_tasks=["constant_slope_linearity"])
```

## New Functions

### `define_plot_tasks()`
Returns a dictionary of all available plot tasks with their configurations.

### `run_plots(plotter, plot_all=True, selected_tasks=None)`
Executes plot tasks with flexible options:
- `plotter`: CadencePlotter instance
- `plot_all`: If True, runs all tasks
- `selected_tasks`: List of specific task names to run

## Class Methods (Unchanged)

All plotting methods remain available:
- `plot_digital_sweep()` - Plot digital sweeps
- `plot_linearity()` - Plot DNL/INL
- `plot_mc_linearity_per_iteration()` - MC linearity analysis
- `plot_pvt_sweep()` - PVT sweeps
- `plot_pvt_linearity()` - PVT linearity
- `plot_histogram()` - Monte Carlo histograms
- `plot_signals()` - Transient signals
- `plot_direct_csv_sweep()` - Direct CSV with decimal codes
- `smart_plot()` - Auto-route to appropriate plot function

## Features

### Direct CSV Sweep Plotting
The `plot_direct_csv_sweep()` function handles CSV files with direct code values:
- Automatically removes code 129 (middle point)
- Re-indexes remaining codes continuously
- Shows sparse decimal labels on x-axis
- Generates both sweep and linearity (DNL/INL) plots

### Sparse X-axis Labels
All plots use sparse x-axis labels (≈10 labels) to avoid crowding when dealing with large code ranges.

### Error Handling
- Individual file errors don't stop the entire run
- Failed tasks are reported with checkmark/X indicators
- Detailed error messages for debugging

## Output Structure

All plots are saved to the configured plot directory organized by subsystem:
- `plots/constant_slope/` - CS plots
- `plots/variable_slope/` - DLCSI plots
- `plots/delayline_csi/` - DLCSI plots
- `plots/Phase_interpolator/` - PI plots

Each plot is named descriptively:
- `sweep_decimal_*.png` - Sweep plots with decimal codes
- `linearity_decimal_*.png` - Linearity (DNL/INL) plots
- `pvt_sweep_*.png` - PVT sweeps
- `pvt_linearity_*.png` - PVT linearity
- etc.

## Customization

### Add New Plot Tasks
Edit `define_plot_tasks()` function in `plot_delay.py`:

```python
"my_new_task": {
    "description": "My custom task description",
    "files": ["file1.csv", "file2.csv"],
    "action": lambda plotter, f: plotter.my_custom_plot(f)
}
```

### Modify File Lists
Update the `"files"` list in any task definition to include/exclude specific CSV files.

### Change Plot Parameters
Modify the parameters in the `"action"` lambda or `"configs"` dict for each task.

## Troubleshooting

### FileNotFoundError
Ensure CSV files exist in `results_cadence/` directory.

### Missing columns
Check that the CSV files have the expected column structure (e.g., ' X', ' Y' for Cadence format).

### No plots generated
Check the console output for error messages. Ensure the plot directory has write permissions.

## Notes

- All original plotting functions are preserved and working
- The refactored code maintains backward compatibility
- Task execution is sequential by default (no parallel processing)
- Failed tasks are logged but don't prevent other tasks from running
