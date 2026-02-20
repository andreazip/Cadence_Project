#!/usr/bin/env python
"""
Quick Start Guide - Publication-Ready Plots

This script demonstrates how to generate presentation and thesis-ready plots
from your CSV data.
"""

from plot_delay import CadencePlotter, run_plots, define_plot_tasks

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def main():
    plotter = CadencePlotter(base_dir="results_cadence")
    
    # ========================================================================
    # OPTION 1: Generate All Plots (Publication Quality)
    # ========================================================================
    print_section("OPTION 1: Generate All Publication-Ready Plots")
    print("Generates all plot types with 300 DPI resolution")
    print("Ready to copy-paste into presentations and papers")
    print("\nCommand:")
    print("  python run_plots_all.py")
    
    # ========================================================================
    # OPTION 2: Generate Specific Plot Types
    # ========================================================================
    print_section("OPTION 2: Generate Specific Plot Types")
    print("Generate only the plots you need:")
    print("""
    linearity           - DNL/INL analysis plots
    digital_sweep       - Code sweep curves
    direct_csv          - Direct CSV with decimal codes
    mc_analysis         - Monte Carlo statistical plots
    pvt_analysis        - Temperature & voltage sweep
    """)
    
    # ========================================================================
    # OPTION 3: Manual Control with Python
    # ========================================================================
    print_section("OPTION 3: Direct Python Control")
    
    # Plot specific files
    print("Generate plots for specific CSV files:\n")
    
    # Linearity plots - Perfect for thesis
    print("1. Linearity Analysis (DNL/INL) - Best for thesis")
    print("   Recommended: Include 2-3 of these in technical sections")
    run_plots(plotter, plot_all=False, selected_tasks=['constant_slope_linearity'])
    
    print("\n2. Direct CSV Sweeps - Best for presentations")
    print("   Recommended: Use as overview slide")
    run_plots(plotter, plot_all=False, selected_tasks=['constant_slope_direct_csv'])
    
    # ========================================================================
    # PLOT QUALITY SPECIFICATIONS
    # ========================================================================
    print_section("PLOT QUALITY SPECIFICATIONS")
    print("""
    Resolution:      300 DPI (publication quality)
    Format:          PNG with transparent background
    Color Space:     RGB (suitable for all media)
    Font:            Times New Roman (serif - professional)
    Line Width:      2.0-2.5pt (visible at any size)
    Markers:         6pt (clear and distinct)
    Grid:            Subtle (alpha=0.3) - enhances readability
    
    Suitable for:
    ✓ PowerPoint presentations
    ✓ Academic papers
    ✓ Journal submissions
    ✓ Thesis documents
    ✓ Technical reports
    ✓ Conference presentations
    """)
    
    # ========================================================================
    # RECOMMENDED WORKFLOW
    # ========================================================================
    print_section("RECOMMENDED WORKFLOW FOR PRESENTATIONS")
    
    print("""
    Step 1: Generate plots
    $ python run_plots_all.py
    
    Step 2: Plots are saved to /plots/ subfolder organized by type:
    - /plots/constant_slope/    → Constant slope circuits
    - /plots/variable_slope/    → Variable slope circuits
    - /plots/Phase_interpolator/→ Phase interpolators
    
    Step 3: Copy to PowerPoint/Sheets
    - Right-click plot image → Copy
    - In PowerPoint → Paste
    - No need for resizing or editing!
    
    Step 4: For LaTeX documents
    \\includegraphics[width=0.8\\textwidth]{path/to/plot.png}
    """)
    
    # ========================================================================
    # RECOMMENDED WORKFLOW FOR THESIS
    # ========================================================================
    print_section("RECOMMENDED WORKFLOW FOR THESIS")
    
    print("""
    Best plots to include:
    
    1. Linearity Analysis (DNL/INL)
       - Shows performance characteristics
       - Demonstrates code monotonicity
       - Include 2-3 different configurations
    
    2. Digital Sweeps
       - Shows delay/power vs code
       - Demonstrates dynamic range & resolution
       - Include main architecture + variants
    
    3. PVT Analysis
       - Shows robustness across conditions
       - Includes temperature & voltage sweeps
       - Professional presentation of operating conditions
    
    4. Monte Carlo Analysis
       - Shows variability and yield
       - Includes histograms and distributions
       - Important for silicon characterization
    
    Typical thesis section includes:
    - 1-2 overview plots (digital sweeps)
    - 2-3 detailed analysis plots (linearity)
    - 1-2 robustness plots (PVT)
    Total: 5-7 plots per circuit variant
    """)
    
    # ========================================================================
    # COPY-PASTE READY
    # ========================================================================
    print_section("PLOTS ARE COPY-PASTE READY")
    print("""
    All generated plots can be directly:
    ✓ Copy-pasted into PowerPoint
    ✓ Embedded in Word documents
    ✓ Included in LaTeX via \\includegraphics
    ✓ Used in web presentations
    ✓ Submitted to journals
    
    NO EDITING REQUIRED - Professional quality out-of-the-box!
    """)
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
