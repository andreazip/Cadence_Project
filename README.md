# Cadence Results Plotter

Automated tool for processing and plotting CSV data exported from Cadence Virtuoso.

## Filename Convention
The plotter identifies the X-axis label and destination folder using:
`[Type]_[Y-Axis]_[X-Axis]_[Bits].csv`

### 1. Folder Routing
- `mc_`: Files containing 'mc' are saved to `/plots/monte_carlo/`.
- `corner_`: Files containing 'corner' are saved to `/plots/corners/`.


### 2. X-Axis Labeling
The plotter automatically extracts the X-axis title from the filename:
- `vs_delay_vdd.csv` → X-axis: **VDD**
- `vs_delay_cap.csv` → X-axis: **Capacitance**
- `vs_delay_mc_tt.csv` → X-axis: **Iteration (Monte Carlo), corner used**
- `vs_delay_T_corner.csv` → X-axis: **Process Corner**

## Key Logic
- **Sanitization**: All signal names (e.g., `/out_DTC`) are converted to safe filenames.
- **Redundancy**: Specifically checks **Code 16** against **Code 0**; if identical, Code 16 is removed to ensure clean linearity plots.
- **5-bit Mode**: Automatically applies $d_4 = \text{NOT}(X)$ if the file is tagged as `5bit`.