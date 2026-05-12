import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import numpy as np
import argparse
from pathlib import Path

from plot_style import apply_science_style

# ============================================================================
# QUICK CLI CHEATSHEET (run from project root)
# ============================================================================
# Quick interactive help:
#   python plot_delay.py jhelp
#
# List plot types:
#   python plot_delay.py --list-types
#
# Auto-detect best plot for a CSV:
#   python plot_delay.py --file cs_delay_code_4bit.csv --type auto
#
# Force linearity (DNL/INL):
#   python plot_delay.py --file cs_delay_code_4bit.csv --type linearity
#
# Common options:
#   --base-dir results_cadence --plot-dir plots
#   --max-realizations 200 --max-iterations 200

apply_science_style()

class CadencePlotter:
    def __init__(self, base_dir="results_cadence", plot_dir=None):
        self.base_dir = Path(base_dir)
        self.plot_dir = Path(plot_dir) if plot_dir else Path("plots")
        self.plot_dir.mkdir(exist_ok=True)
        
        # Professional color palette for consistent styling
        self.colors = {
            'primary': '#D62728',      # Crimson Red
            'secondary': '#1F77B4',    # Steel Blue  
            'tertiary': '#2CA02C',     # Forest Green
            'quaternary': '#FF7F0E',   # Dark Orange
            'accent': '#9467BD',       # Purple
            'neutral': '#7F7F7F',      # Gray
        }

    # ========================================================================
    # HELPER METHODS - Reduce Redundancy
    # ========================================================================
    
    def _save_figure(self, fig, save_path, dpi=300, bbox_inches='tight'):
        """Unified figure saving with consistent parameters."""
        fig.canvas.draw()  # Ensure figure is fully rendered
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=0.05)
        print(f"  Saved: {save_path.name}")
        plt.close(fig)  # Close specific figure
    
    def _create_figure(self, nrows=1, ncols=1, figsize=None, sharex=False, sharey=False):
        """Unified figure creation with consistent parameters."""
        if figsize is None:
            figsize = (10, 6) if nrows == 1 else (10, 5*nrows)
        
        if nrows == 1 and ncols == 1:
            fig, ax = plt.subplots(figsize=figsize)
            return fig, ax
        else:
            return plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    
    def _apply_grid_styling(self, ax, alpha=0.6):
        """Apply consistent grid styling."""
        ax.grid(True, alpha=alpha, linestyle='--', linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)

    def _get_marker_cycle(self):
        return ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*', 'h']

    def _should_remove_redundant_code(self, filename):
        """Return True when a dataset contains a known redundant code."""
        return "counter" in Path(filename).name.lower()

    def _remove_redundant_code(self, plot_df, bit_count, filename):
        """Remove the redundant middle code and re-index labels if needed."""
        if plot_df is None or plot_df.empty:
            return plot_df
        if not self._should_remove_redundant_code(filename):
            return plot_df

        remove_idx = len(plot_df) // 2 + 1
        if remove_idx < len(plot_df):
            plot_df = plot_df.drop(plot_df.index[remove_idx]).reset_index(drop=True)

        plot_df['code_index'] = np.arange(len(plot_df))
        if 'label' in plot_df.columns:
            plot_df['label_index'] = plot_df['code_index'].apply(
                lambda c: bin(int(c))[2:].zfill(bit_count)
            )

        return plot_df
    
    def _format_title(self, text):
        """Remove underscores and format title text for display."""
        if not text:
            return text
        # Replace underscores with spaces and title case
        return str(text).replace('_', ' ').replace('-', ' ')
    
    def _format_plot_labels(self, ax, xlabel=None, ylabel=None, title=None):
        """Apply consistent label formatting."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    def _get_metric_info(self, filename, metric_col=None):
        """Get metric type info (power vs delay) and scaling."""
        is_power = any(x in filename.lower() for x in ['power', 'p_avg', 'pinst'])
        
        if is_power:
            return {
                'is_power': True,
                'scale': 1e6,
                'prefix': 'μ',
                'unit': 'W',
                'name': 'Power'
            }
        else:
            return {
                'is_power': False,
                'scale': 1e9,
                'prefix': 'n',
                'unit': 's',
                'name': 'Delay'
            }

    def _sanitize(self, name):
        """Replaces characters like '/' or '\' that break file paths."""
        return re.sub(r'[\\/*?:"<>|]', '_', str(name)).strip('_')
    
    def load_data(self, filename):
        path = Path(filename)
        if path.suffix == "":
            path = path.with_suffix(".csv")
        if not path.exists():
            path = self.base_dir / filename
            if path.suffix == "":
                path = path.with_suffix(".csv")
        if not path.exists():
            print(f"Warning: File {filename} not found.")
            return None, path
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df, path

    def _compute_dnl_inl(self, y, lsb=None):
        """Compute DNL/INL robustly and return (dnl, inl, lsb_used)."""
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) < 2:
            return np.array([]), np.array([]), 0.0

        if lsb is None:
            lsb = (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)

        if lsb == 0 or np.isnan(lsb):
            return np.array([]), np.array([]), lsb

        dnl = np.insert(np.diff(y_arr) / lsb - 1, 0, 0)
        inl = np.cumsum(dnl)
        return dnl, inl, lsb

    def _remove_middle_code_rows(self, data):
        """Remove redundant middle code row from column-based MC matrices."""
        if data is None or data.shape[0] == 0:
            return data, None
        mid_idx = data.shape[0] // 2
        keep_indices = [i for i in range(data.shape[0]) if i != mid_idx]
        return data.iloc[keep_indices, :], mid_idx

    def _get_axis_labels(self, filename):
        """Extracts the X-axis parameter from filenames like 'vs_delay_vdd.csv'."""
        fname = Path(filename).stem.lower()
        parts = fname.split('_')
        # Convention: [type]_[y]_[x] -> parts[2] is usually the X axis
        if len(parts) >= 3:
            x_label = parts[2].replace('code', 'Digital Code').capitalize()
        else:
            x_label = "Parameter"
        return x_label

    def _get_sweep_xlabel(self, x_values):
        """Infer a readable X-axis label for sweep plots."""
        if x_values is None or len(x_values) == 0:
            return "Sweep Parameter"
        try:
            x_arr = np.asarray(x_values, dtype=float)
        except (TypeError, ValueError):
            return "Sweep Parameter"

        if np.allclose(x_arr, np.round(x_arr), rtol=0, atol=1e-9):
            return "Digital Code"
        return "Sweep Parameter"
    
    def _get_thesis_title(self, filename, plot_type="linearity"):
        """Generate professional, thesis-ready titles from filenames.
        
        Args:
            filename: The data filename
            plot_type: Type of plot (linearity, sweep, mc_linearity, mc_sweep, corner, etc.)
        
        Returns:
            A professional title string
        """
        fname = Path(filename).stem.lower()
        
        # Extract circuit type
        if fname.startswith('cs'):
            circuit = "Constant-Slope"
        elif fname.startswith('vs'):
            circuit = "Variable-Slope"
        elif fname.startswith('dlcsi'):
            circuit = "Delay-Line CSI"
        elif fname.startswith('pi'):
            circuit = "Phase Interpolator"
        else:
            circuit = "ADC"
        
        # Determine bit depth and configuration
        bit_match = re.search(r'(\d)bit', fname)
        bits = bit_match.group(1) if bit_match else "8"
        
        # Check for special configurations
        if 'counter' in fname:
            config = "Thermometric Counter"
        elif 'coarse' in fname:
            config = "Coarse"
        elif 'fine' in fname:
            config = "Fine"
        elif 'coarse-fine' in fname:
            config = "Coarse-Fine"
        elif 'nonidealcurs' in fname:
            config = "Non-Ideal Current Sources"
        elif 'mc' in fname:
            config = "Monte Carlo"
        else:
            config = "Standard"
        
        # Determine metric (delay vs power)
        metric = "Power" if 'power' in fname else "Delay"
        
        # Generate title based on plot type
        if plot_type == "linearity":
            return f"{circuit} - {metric} Linearity ({bits}-Bit {config})"
        elif plot_type == "sweep":
            return f"{circuit} - {metric} vs Code ({bits}-Bit {config})"
        elif plot_type == "mc_linearity_all":
            num = "200" if "counter_mc" in fname else "100"
            return f"{circuit} - DNL/INL Analysis ({bits}-Bit, {num} Realizations)"
        elif plot_type == "mc_sweep_all":
            num = "200" if "counter_mc" in fname else "100"
            return f"{circuit} - {metric} vs Code ({bits}-Bit, {num} Realizations)"
        elif plot_type == "mc_distribution":
            return f"{circuit} - {metric} Distribution (Monte Carlo)"
        elif plot_type == "corner":
            # Extract corner info
            if 'vdd' in fname:
                return f"{circuit} - {metric} vs Supply Voltage"
            elif 'temp' in fname or 'temperature' in fname:
                return f"{circuit} - {metric} vs Temperature"
            else:
                return f"{circuit} - {metric} (PVT Analysis)"
        elif plot_type == "signal":
            return f"{circuit} - Transient Signals"
        else:
            return f"{circuit} - {metric} Analysis ({bits}-Bit)"
    
    
    def _get_save_path(self, filename, signal_name, plot_type):
        """Routes files to 'variable_slope' or 'constant_slope' subfolders."""
        subfolder = "other"
        fname_lower = Path(filename).name.lower()
        
        if fname_lower.startswith("vs"):
            subfolder = "variable_slope"
        elif fname_lower.startswith("cs"):
            subfolder = "constant_slope"
        elif fname_lower.startswith("dlcsi"):
            subfolder = "delayline_csi"
        elif fname_lower.startswith("pi"):
            subfolder = "Phase_interpolator"
        
        # Create the subfolder path
        target_dir = self.plot_dir / subfolder
        target_dir.mkdir(exist_ok=True)
        
        # Sanitize parts to build safe filename
        safe_sig = self._sanitize(signal_name)
        safe_stem = self._sanitize(Path(filename).stem)

        return target_dir / f"{plot_type}_{safe_stem}.png"
    
    def _get_bit_info(self, filename, df):
        """Detects bit depth from filename or columns."""
        # Try filename first
        match = re.search(r"(\d)bit", filename.lower())
        if match:
            return int(match.group(1))
        
        # Fallback: Count unique bit indices in headers (d0, d1, b2...)
        found_bits = set(re.findall(r"[db](\d+)=", "".join(df.columns)))
        if found_bits:
            return len(found_bits) + 1 # +1 for the sweep bit
        return 3 # Default

    def _filter_mc_iterations(self, df):
        return df

    def _reconstruct_mc_iteration_data(self, df, filename, bit_count, metric_col):
        """Reconstruct codes for a single MC iteration without averaging."""
        df = self._filter_mc_iterations(df)

        bit_cols = sorted([c for c in df.columns if c.startswith('d') and c[1:].isdigit()])
        if len(bit_cols) < bit_count:
            return None

        df_clean = df[bit_cols + [metric_col]].copy()
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

        is_cs = Path(filename).name.lower().startswith("cs")
        codes = np.zeros(len(df_clean), dtype=int)
        for bit_col in bit_cols:
            bit_idx = int(bit_col[1:])
            bit_vals = df_clean[bit_col].astype(int).values
            if is_cs and bit_count == 5 and bit_idx == 4:
                bit_vals = 1 - bit_vals
            codes = codes | (bit_vals << bit_idx)

        df_clean['code'] = codes
        df_clean = df_clean.sort_values('code').drop_duplicates('code').reset_index(drop=True)

        is_counter = "counter" in Path(filename).name.lower()
        if is_cs and not is_counter:
            if 15 in df_clean['code'].values:
                df_clean = df_clean[df_clean['code'] != 15].reset_index(drop=True)

        df_clean['label'] = df_clean['code'].apply(lambda c: bin(int(c))[2:].zfill(bit_count))
        df_clean['code_index'] = np.arange(len(df_clean))
        df_clean['label_index'] = df_clean['code_index'].apply(lambda c: bin(int(c))[2:].zfill(bit_count))
        df_clean['y'] = df_clean[metric_col]

        return self._remove_redundant_code(df_clean, bit_count, filename)

    def _reconstruct_mc_code_sweep(self, df, filename, bit_count):
        """Reconstruct digital codes from individual bit columns in Monte Carlo data.
        Format: Columns like d0, d1, d2, d3, d4 with a 'delay' or 'P_avg' column."""
        df = self._filter_mc_iterations(df)
        # Identify the metric column (delay or power)
        metric_col = None
        for col in ['delay', 'power', 'P_avg']:
            if col in df.columns:
                metric_col = col
                break
        if metric_col is None:
            return None
        
        # Find all bit columns (d0, d1, d2, d3, d4)
        bit_cols = sorted([c for c in df.columns if c.startswith('d') and c[1:].isdigit()])
        if len(bit_cols) < bit_count:
            return None
        
        # Reconstruct codes from bits
        df_clean = df[bit_cols + [metric_col]].copy()
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Calculate binary codes using numpy operations
        is_cs = Path(filename).name.lower().startswith("cs")
        codes = np.zeros(len(df_clean), dtype=int)
        for bit_col in bit_cols:
            bit_idx = int(bit_col[1:])
            bit_vals = df_clean[bit_col].astype(int).values
            if is_cs and bit_count == 5 and bit_idx == 4:
                bit_vals = 1 - bit_vals
            codes = codes | (bit_vals << bit_idx)
        
        df_clean['code'] = codes
        
        # Group by code and calculate statistics
        data = []
        for code in sorted(np.unique(codes)):
            group_data = df_clean[df_clean['code'] == code][metric_col]
            if len(group_data) == 0:
                continue
            
            # Average across MC iterations
            y_val = group_data.mean()
            label = bin(int(code))[2:].zfill(bit_count)
            data.append({'code': int(code), 'y': y_val, 'label': label})
        
        if not data:
            return None
        
        plot_df = pd.DataFrame(data).sort_values('code').drop_duplicates('code').reset_index(drop=True)
        
        # --- Remove Code 15 ONLY if it is a constant slope (cs) file ---
        if is_cs:
            if 15 in plot_df['code'].values:
                plot_df = plot_df[plot_df['code'] != 15].reset_index(drop=True)

        return self._remove_redundant_code(plot_df, bit_count, filename)

    def _reconstruct_digital_data(self, df, filename, signal_name, bit_count):
        """Generic reconstruction for any bit depth with PI thermometer support."""
        df = self._filter_mc_iterations(df)
        data = []
        
        # --- STRICT FILENAME CHECK ---
        is_cs = Path(filename).name.lower().startswith("cs")
        is_pi = Path(filename).name.lower().startswith("pi")
        
        # Find all columns for this signal
        x_cols = [c for c in df.columns if c.startswith(signal_name) and (c.endswith(' X') or c.endswith('X'))]
        if not x_cols:
            return None

        for x_col in x_cols:
            # Handle variations in whitespace for Cadence headers
            y_col = x_col[:-1] + 'Y' if x_col.endswith('X') else x_col[:-2] + ' Y'
            if y_col not in df.columns: continue

            # regex includes 's' bits for PI thermometer codes
            header_bits = re.findall(r"([dbs])(\d+)=(\d+)", x_col)
            
            bit_dict = {}
            for prefix, b_idx, b_val in header_bits:
                bit_dict[int(b_idx)] = int(b_val)
            
            all_indices = set(range(bit_count))
            found_indices = set(bit_dict.keys())
            missing = sorted(list(all_indices - found_indices))
            sweep_bit_idx = missing[0] if missing else bit_count - 1

            temp = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
            for _, row in temp.iterrows():
                x_val = int(row[x_col])
                
                # Apply inversion logic ONLY if it is a constant slope (cs) 5-bit file
                if is_cs and bit_count == 5 and sweep_bit_idx == 4:
                    d_sweep = 1 - x_val
                else:
                    d_sweep = x_val

                # Construct full bit state for this data point
                current_bits = bit_dict.copy()
                current_bits[sweep_bit_idx] = d_sweep
                
                # DETERMINE CODE AND LABEL
                if is_pi:
                    # For PI, code is the sum of bits (thermometer level)
                    code = sum(current_bits.values())
                    # Construct label string directly (MSB to LSB)
                    label = "".join(str(current_bits.get(i, 0)) for i in range(bit_count-1, -1, -1))
                else:
                    # Standard binary weighted logic
                    code = 0
                    for idx, val in current_bits.items():
                        code |= (val << idx)
                    label = bin(code)[2:].zfill(bit_count)

                data.append({'code': code, 'y': row[y_col], 'label': label})
        
        if not data: return None
        plot_df = pd.DataFrame(data).sort_values('code').drop_duplicates('code').reset_index(drop=True)
        
        # --- Remove Code 15 ONLY if it is a constant slope (cs) file ---
        if is_cs:
            if 15 in plot_df['code'].values:
                plot_df = plot_df[plot_df['code'] != 15].reset_index(drop=True)

        plot_df['code_index'] = np.arange(len(plot_df))
        plot_df['label_index'] = plot_df['code_index'].apply(lambda c: bin(int(c))[2:].zfill(bit_count))
        
        return plot_df

    def _extract_mcparamset_sweeps(self, df):
        """Extract sweep data for files with mcparamset X/Y columns."""
        x_cols = [c for c in df.columns if c.endswith(' X') and 'mcparamset' in c.lower()]
        sweeps = []

        for x_col in x_cols:
            y_col = x_col[:-2] + ' Y'
            if y_col not in df.columns:
                continue

            match = re.search(r"mcparamset=(\d+)", x_col)
            iteration = int(match.group(1)) if match else len(sweeps) + 1

            x_vals = pd.to_numeric(df[x_col], errors='coerce')
            y_vals = pd.to_numeric(df[y_col], errors='coerce')
            mask = x_vals.notna() & y_vals.notna()
            if not mask.any():
                continue

            x_clean = x_vals[mask].values
            y_clean = y_vals[mask].values

            # Always remove elements at indices 0 and 129 if present.
            drop_indices = [0, 128]
            keep_mask = np.ones(len(x_clean), dtype=bool)
            for idx in drop_indices:
                if idx < len(keep_mask):
                    keep_mask[idx] = False

            x_clean = x_clean[keep_mask]
            y_clean = y_clean[keep_mask]

            sweeps.append({
                "iteration": iteration,
                "x": x_clean,
                "y": y_clean,
            })

        return sorted(sweeps, key=lambda s: s["iteration"])

    def plot_mcparamset_sweep(self, filename, max_realizations=200):
        """Plot sweep curves for mcparamset X/Y data with mean overlay."""
        df, path = self.load_data(filename)
        if df is None:
            return

        sweeps = self._extract_mcparamset_sweeps(df)
        if not sweeps:
            return

        sweeps = sweeps[:max_realizations]
        metric_info = self._get_metric_info(filename)

        fig, ax = self._create_figure(figsize=(12, 7))

        for idx, sweep in enumerate(sweeps):
            x_vals = sweep["x"]
            y_vals = sweep["y"]
            if len(y_vals) == 0:
                continue

            y_to_plot = (y_vals - y_vals[0]) * metric_info['scale']
            ax.plot(x_vals, y_to_plot,
                    color=self.colors['primary'], alpha=0.2, linewidth=1.4)

        xlabel = self._get_sweep_xlabel(sweeps[0]["x"])
        thesis_title = self._get_thesis_title(filename, "mc_sweep_all")
        clean_title = self._format_title(thesis_title)

        self._format_plot_labels(
            ax,
            xlabel=xlabel,
            ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
            title=clean_title,
        )
        self._apply_grid_styling(ax, alpha=0.35)

        plt.tight_layout()
        save_path = self._get_save_path(filename, "MC_Sweep", f"mcparamset_sweep_{len(sweeps)}")
        self._save_figure(fig, save_path)

    def plot_mcparamset_linearity(self, filename, max_realizations=200):
        """Plot DNL/INL for mcparamset X/Y data with mean overlay."""
        df, path = self.load_data(filename)
        if df is None:
            return

        sweeps = self._extract_mcparamset_sweeps(df)
        if not sweeps:
            return

        sweeps = sweeps[:max_realizations]

        fig, (ax_dnl, ax_inl) = self._create_figure(nrows=2, ncols=1, figsize=(12, 9), sharex=True)

        for idx, sweep in enumerate(sweeps):
            y_vals = sweep["y"]
            if len(y_vals) < 2:
                continue

            dnl, inl, _ = self._compute_dnl_inl(y_vals)
            if len(dnl) == 0:
                continue
            x_idx = np.arange(len(y_vals))

            ax_dnl.plot(x_idx, dnl, color=self.colors['secondary'], alpha=0.2, linewidth=1.4)
            ax_inl.plot(x_idx, inl, color=self.colors['primary'], alpha=0.2, linewidth=1.4)

        thesis_title = self._get_thesis_title(filename, "mc_linearity_all")
        clean_title = self._format_title(thesis_title)

        ax_dnl.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
        ax_dnl.set_title(clean_title, fontsize=14, fontweight='bold', pad=15)
        ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_dnl.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='±0.5 LSB')
        ax_dnl.axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax_dnl.set_ylim(-0.7, 0.7)
        ax_dnl.legend(fontsize=10, loc='upper right', framealpha=0.95)
        self._apply_grid_styling(ax_dnl, alpha=0.35)

        ax_inl.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
        ax_inl.set_title("Integral Non-Linearity (INL)", fontsize=14, fontweight='bold', pad=15)
        ax_inl.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
        ax_inl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_inl.legend(fontsize=10, loc='upper right', framealpha=0.95)
        self._apply_grid_styling(ax_inl, alpha=0.35)

        plt.tight_layout()
        save_path = self._get_save_path(filename, "MC_Linearity", f"mcparamset_linearity_{len(sweeps)}")
        self._save_figure(fig, save_path)

    def plot_average_power_by_code(self, filename, start_time=2e-8, window_size=5e-8, num_codes=256, remove_code=True, P_static = 0):
        """Plot average power consumption by digital code.
        
        Uses sliding window analysis to calculate average power for each code. By default,
        the middle code is removed for cleaner visualization.
        The plotted power is offset-corrected by subtracting static power at code 0.
        
        Args:
            filename: CSV file with transient power data (Time, Power columns)
            start_time: Start time for the analysis window (default: 2e-8 s)
            window_size: Duration of each code's time window (default: 5e-8 s)
            num_codes: Total number of codes to process (default: 256)
            remove_code: If True, remove the middle code index and shift right side left by one
            P_static: Static power to subtract (default: 0)
        """
        df, path = self.load_data(filename)
        if df is None:
            return

        # Identify time and power columns
        time_col = df.columns[0]
        power_col = df.columns[1]
        df = df.rename(columns={time_col: 'Time', power_col: 'Power'})
        
        # Calculate average power for each code using windowed integration
        all_averages = []
        all_codes = []
        
        for i in range(num_codes):
            t_start = start_time + i * window_size
            t_end = start_time + (i + 1) * window_size
            
            mask = (df['Time'] >= t_start) & (df['Time'] < t_end)
            window_data = df[mask]
            
            if not window_data.empty:
                times = window_data['Time'].values
                powers = np.abs(window_data['Power'].values)
                # Trapezoidal integration: sum of (dx * (y[i] + y[i+1])/2)
                energy = np.sum((times[1:] - times[:-1]) * (powers[1:] + powers[:-1]) / 2.0)
                actual_duration = times[-1] - times[0]
                avg_p = energy / actual_duration if actual_duration > 0 else window_data['Power'].mean()
            else:
                avg_p = 0
            
            all_averages.append(avg_p)
            all_codes.append(i)
        
        # Static power offset from code 0.
        ##static_power_w = float(all_averages[0]) if len(all_averages) > 0 else 0.0

        # Remove middle code and shift all right-side codes one step left.
        remove_idx = num_codes // 2 +1 if num_codes > 0 else None

        plot_averages = []
        if remove_code and remove_idx is not None:
            for i in range(num_codes):
                if i == remove_idx:
                    continue
                plot_averages.append(all_averages[i])
            # Re-index to contiguous codes after removal (right half shifts left by one).
            plot_codes = list(range(len(plot_averages)))
        else:
            plot_averages = list(all_averages)
            plot_codes = list(all_codes)

        # Remove static power offset for plotting.
        plot_averages_offset_w = np.array(plot_averages)  - P_static
        # Create publication-ready plot
        fig, ax = self._create_figure(figsize=(12, 7))
        
        legend_label = (
            f"Average Power - $P_{{static}}$, $P_{{static}}$={P_static * 1e6:.3f} uW"
        )
        if remove_code and remove_idx is not None:
            legend_label += f", removed code {remove_idx}"

        ax.plot(plot_codes, plot_averages_offset_w * 1e6, color=self.colors['primary'],
                linewidth=2.6, marker=None, label=legend_label)
        
        thesis_title = self._get_thesis_title(filename, "sweep")
        clean_title = self._format_title(thesis_title)
        
        self._format_plot_labels(
            ax,
            xlabel="Digital Code",
            ylabel="Average Power (µW)",
            title=clean_title
        )
        
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        self._apply_grid_styling(ax, alpha=0.35)
        
        plt.tight_layout()
        save_path = self._get_save_path(filename, "Avg_Power", "avg_power_by_code")
        self._save_figure(fig, save_path)

    def plot_digital_sweep(self, filename, signal_name=None, max_iterations=200):
        """Plot digital code sweeps with improved styling and metrics."""
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        if signal_name is None:
            signal_name = df.columns[0].split(' (')[0].split(' ')[0]

        # Check if this is MC code sweep format
        has_bit_cols = all(f'd{i}' in df.columns for i in range(bit_count))
        has_x_y_cols = any(c.endswith(' X') or c.endswith('X') for c in df.columns)
        has_mcparamset = any('mcparamset' in c.lower() for c in df.columns)

        if has_x_y_cols and has_mcparamset:
            return self.plot_mcparamset_sweep(filename, max_realizations=max_iterations)
        
        # MC per-iteration plotting
        if has_bit_cols and not has_x_y_cols and 'mc_iteration' in df.columns:
            df = self._filter_mc_iterations(df)
            metric_col = next((c for c in ['delay', 'power', 'P_avg'] if c in df.columns), None)
            if metric_col is None:
                return

            iterations = sorted(df['mc_iteration'].unique())[:max_iterations]
            metric_info = self._get_metric_info(signal_name)
            
            fig, ax = self._create_figure(figsize=(11, 6))
            
            for idx, iteration in enumerate(iterations):
                df_iter = df[df['mc_iteration'] == iteration].copy()
                plot_df = self._reconstruct_mc_iteration_data(df_iter, filename, bit_count, metric_col)
                if plot_df is None:
                    continue
                
                y = plot_df['y'].values
                x_labels = plot_df['label_index'] if 'label_index' in plot_df.columns else plot_df['label']
                y_to_plot = (y - y[0]) * metric_info['scale'] if not metric_info['is_power'] else y * metric_info['scale']
                
                # Plot all 200 with high transparency, no labels
                ax.plot(x_labels, y_to_plot, linewidth=1.5,
                        alpha=0.2, color=self.colors['primary'])
            
            self._format_plot_labels(ax,
                                    xlabel=f"Digital Code",
                                    ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
                                    title=f"{bit_count}-Bit Monte Carlo Sweep - All {len(iterations)} Iterations - {path.stem.replace('_', ' ').title()}"
            )
            ax.tick_params(axis='x', rotation=45)
            self._apply_grid_styling(ax)
            
            plt.tight_layout()
            save_path = self._get_save_path(filename, signal_name, f"sweep_mc_all_{len(iterations)}_iterations")
            self._save_figure(fig, save_path)
            return

        # Standard digital sweep
        if has_bit_cols and not has_x_y_cols:
            plot_df = self._reconstruct_mc_code_sweep(df, filename, bit_count)
        else:
            plot_df = self._reconstruct_digital_data(df, filename, signal_name, bit_count)
        
        if plot_df is None:
            return

        y = plot_df['y'].values
        metric_info = self._get_metric_info(signal_name)
        
        # Calculate metrics
        y_to_plot = (y - y[0]) * metric_info['scale'] if not metric_info['is_power'] else y * metric_info['scale']
        dr = (y.max() - y.min()) * metric_info['scale']
        res = dr / (len(y) - 1) if len(y) > 1 else 0

        # Plot
        fig, ax = self._create_figure(figsize=(11, 6))
        ax.plot(plot_df['label'], y_to_plot, marker=None, color=self.colors['primary'], 
            linewidth=2.6, 
                label=f"DR: {dr:.3f} {metric_info['prefix']}{metric_info['unit']} | "
                      f"Res: {res*1000:.2f} p{metric_info['unit']}")
        
        thesis_title_sweep = self._get_thesis_title(filename, "sweep")
        clean_title = self._format_title(thesis_title_sweep)
        self._format_plot_labels(ax,
                                xlabel=f"Digital Code",
                                ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
                                title=clean_title
        )
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=10, loc='best')
        self._apply_grid_styling(ax)

        plt.tight_layout()
        save_path = self._get_save_path(filename, signal_name, "sweep")
        self._save_figure(fig, save_path)

    def plot_mc_sweep_all_realizations(self, filename, max_realizations=200):
        """
        Plot delay vs code for all MC realizations overlaid with improved styling.
        Format: Each column is one iteration's delay data (256 values)
        - Skip first column  
        - Remove first row (code 0) and row 128 (redundant code)
        """
        df = pd.read_csv(self.base_dir / filename, header=None)
        
        if df.shape[0] < 2:
            print(f"Error: File {filename} has insufficient rows")
            return
        
        # Skip first column and every other column (alternating columns are zeros)
        data = df.iloc[1:, 1::2].astype(float)  # Skip header row, first column, and every other column
        
        num_realizations = min(data.shape[1], max_realizations)
        num_codes = data.shape[0]
        
        data_subset = data.iloc[:, :num_realizations]
        data_clean, removed_idx = self._remove_middle_code_rows(data_subset)
        codes = np.arange(data_clean.shape[0])
        
        print(f"Processing {num_realizations} sweep curves with {data_clean.shape[0]} codes per curve")
        if removed_idx is not None:
            print(f"Removed redundant middle code at index {removed_idx}")
        
        # Create figure
        fig, ax = self._create_figure(figsize=(12, 7))
        
        for iter_idx in range(num_realizations):
            delays = data_clean.iloc[:, iter_idx].values
            
            # Normalize to relative delay (subtract first value)
            delays_relative = (delays - delays[0]) * 1e9  # Convert to ns
            
            # Plot with transparency and subtle styling
            ax.plot(codes, delays_relative,
                     color=self.colors['primary'], alpha=0.2, linewidth=1.3)
        
        # Add mean line
        
        # Formatting
        ax.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
        ax.set_ylabel("Relative Delay (ns)", fontsize=12, fontweight='bold')
        thesis_title = self._get_thesis_title(filename, "mc_sweep_all")
        clean_title = self._format_title(thesis_title)
        ax.set_title(clean_title, fontsize=14, fontweight='bold', pad=15)
        self._apply_grid_styling(ax)
        
        plt.tight_layout()
        
        save_path = self._get_save_path(filename, "MC_Sweep", f"mc_all_{num_realizations}_sweeps")
        self._save_figure(fig, save_path)

    def plot_mc_linearity_all_realizations_xy(self, filename, max_realizations=200):
        """
        Plot DNL and INL for MC data with column-based iterations (improved styling).
        Format: Each column is one iteration's delay data (256 values)
        - Skip first column
        - Remove first row (code 0) and rows 129-130 (redundant codes)
        - Remaining: 256 curves with 253 points each (or adjusted after removal)
        """
        df = pd.read_csv(self.base_dir / filename, header=None)
        
        if df.shape[0] < 2:
            print(f"Error: File {filename} has insufficient rows")
            return
        
        # Skip first column and every other column (alternating columns are zeros)
        data = df.iloc[1:, 1::2].astype(float)  # Skip header row, first column, and every other column
        
        num_realizations = min(data.shape[1], max_realizations)
        num_codes = data.shape[0]
        
        data_subset = data.iloc[:, :num_realizations]
        data_clean, removed_idx = self._remove_middle_code_rows(data_subset)

        print(f"Processing {num_realizations} iterations with {data_clean.shape[0]} codes per iteration")
        if removed_idx is not None:
            print(f"Removed redundant middle code at index {removed_idx}")
        
        # Create figure with 2 subplots (DNL and INL)
        fig, (ax_dnl, ax_inl) = self._create_figure(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        
        for iter_idx in range(num_realizations):
            delays = data_clean.iloc[:, iter_idx].values
            codes = np.arange(len(delays))
            
            if len(delays) < 2:
                continue

            dnl, inl, _ = self._compute_dnl_inl(delays)
            if len(dnl) == 0:
                continue
            
            # Plot with subtle styling
            ax_dnl.plot(codes, dnl,
                       color=self.colors['secondary'], alpha=0.2, linewidth=1.2)
            ax_inl.plot(codes, inl,
                       color=self.colors['primary'], alpha=0.2, linewidth=1.2)
        
        # Formatting
        thesis_title_dnl = self._get_thesis_title(filename, "mc_linearity_all")
        clean_title = self._format_title(thesis_title_dnl)
        
        ax_dnl.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
        ax_dnl.set_title(clean_title, fontsize=14, fontweight='bold', pad=15)
        self._apply_grid_styling(ax_dnl, alpha=0.35)
        ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_dnl.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='±0.5 LSB')
        ax_dnl.axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax_dnl.set_ylim(-0.7, 0.7)
        ax_dnl.legend(fontsize=10, loc='upper right', framealpha=0.95)
        
        ax_inl.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
        ax_inl.set_title("Integral Non-Linearity (INL)", fontsize=14, fontweight='bold', pad=15)
        ax_inl.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
        self._apply_grid_styling(ax_inl, alpha=0.35)
        ax_inl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_inl.legend(fontsize=10, loc='upper right', framealpha=0.95)

        plt.tight_layout()

        save_path = self._get_save_path(filename, "MC_Linearity", f"mc_all_{num_realizations}_dnl_inl")
        self._save_figure(fig, save_path)

    def plot_mc_linearity_per_iteration(self, filename, lsb_ns=None, max_iterations=200):
        """
        Plot DNL and INL for each individual MC realization.
        
        Args:
            filename: MC data file
            lsb_ns: Resolution in nanoseconds (default: 0.3 ns)
        """
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        lsb = lsb_ns * 1e-9 if lsb_ns is not None else None
        
        # Find metric column
        metric_col = None
        for col in ['delay', 'power', 'P_avg']:
            if col in df.columns:
                metric_col = col
                break
        if metric_col is None: return
        
        # Find bit columns
        bit_cols = sorted([c for c in df.columns if c.startswith('d') and c[1:].isdigit()])
        if len(bit_cols) < bit_count:
            return
        
        # Check for mc_iteration column
        if 'mc_iteration' not in df.columns:
            print(f"Warning: No 'mc_iteration' column in {filename}")
            return

        df = self._filter_mc_iterations(df)
        
        # Create figure with 2 subplots (DNL and INL)
        fig, (ax_dnl, ax_inl) = self._create_figure(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
        
        # Iterate through each MC realization
        iterations = sorted(df['mc_iteration'].unique())[:max_iterations]
        num_iterations = len(iterations)
        
        for idx, iteration in enumerate(iterations):
            df_iter = df[df['mc_iteration'] == iteration].copy()
            plot_df = self._reconstruct_mc_iteration_data(df_iter, filename, bit_count, metric_col)
            if plot_df is None:
                continue

            y = plot_df['y'].values
            x_labels = plot_df['label_index'] if 'label_index' in plot_df.columns else plot_df['label']
            
            # Calculate DNL and INL
            if len(y) > 1:
                dnl, inl, _ = self._compute_dnl_inl(y, lsb=lsb)
                if len(dnl) == 0:
                    continue
                
                # Plot with transparency - NO LABELS to avoid cluttering
                ax_dnl.plot(x_labels, dnl,
                           color=self.colors['secondary'], alpha=0.2, linewidth=1.2)
                ax_inl.plot(x_labels, inl,
                           color=self.colors['primary'], alpha=0.2, linewidth=1.2)
        
        # Formatting
        ax_dnl.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
        ax_dnl.set_title(f"DNL - All {num_iterations} MC Realizations (LSB=ideal)", fontsize=14, fontweight='bold')
        self._apply_grid_styling(ax_dnl)
        ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_dnl.axhline(y=1, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='±1 LSB')
        ax_dnl.axhline(y=-1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_dnl.legend(fontsize=10, loc='upper right')
        
        ax_inl.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
        ax_inl.set_title(f"INL - All {num_iterations} MC Realizations (LSB=ideal)", fontsize=14, fontweight='bold')
        ax_inl.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
        self._apply_grid_styling(ax_inl)
        ax_inl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

        plt.tight_layout()

        save_path = self._get_save_path(filename, "MC_Linearity", f"mc_all_{num_iterations}_iterations")
        self._save_figure(fig, save_path)

    def plot_linearity(self, filename, signal_name=None):
        """Plot DNL and INL with publication-ready styling."""
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        if signal_name is None:
            signal_name = df.columns[0].split(' (')[0].split(' ')[0]

        # Check if this is MC code sweep format
        has_bit_cols = all(f'd{i}' in df.columns for i in range(bit_count))
        has_x_y_cols = any(c.endswith(' X') or c.endswith('X') for c in df.columns)
        has_mcparamset = any('mcparamset' in c.lower() for c in df.columns)

        if has_x_y_cols and has_mcparamset:
            return self.plot_mcparamset_linearity(filename)
        
        if has_bit_cols and not has_x_y_cols:
            plot_df = self._reconstruct_mc_code_sweep(df, filename, bit_count)
        else:
            plot_df = self._reconstruct_digital_data(df, filename, signal_name, bit_count)
        
        if plot_df is None: return

        y = plot_df['y'].values
        dnl, inl, lsb_ideal = self._compute_dnl_inl(y)
        if len(dnl) == 0:
            return

        # Create figure with improved styling
        fig, (ax1, ax2) = self._create_figure(nrows=2, ncols=1, figsize=(11, 9), sharex=True)
        
        # DNL plot
        ax1.plot(plot_df['label'], dnl, marker=None, color=self.colors['secondary'], 
                 linewidth=2.6, label='DNL', alpha=0.9)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax1.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        thesis_title = self._get_thesis_title(filename, "linearity")
        clean_title = self._format_title(thesis_title)
        self._format_plot_labels(ax1, ylabel='DNL (LSB)', title=clean_title)
        self._apply_grid_styling(ax1, alpha=0.35)
        ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
        
        # INL plot
        ax2.plot(plot_df['label'], inl, marker=None, color=self.colors['primary'], 
                 linewidth=2.6, label='INL', alpha=0.9)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        self._format_plot_labels(ax2, xlabel=f'Digital Code (LSB={lsb_ideal:.2e})', 
                                ylabel='INL (LSB)')
        self._apply_grid_styling(ax2, alpha=0.35)
        ax2.legend(fontsize=11, loc='upper right', framealpha=0.95)
        
        # Rotate x-axis labels for readability
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self._get_save_path(filename, signal_name, "linearity")
        self._save_figure(fig, save_path)

    def plot_direct_csv_sweep(self, filename, y_col_idx=1, remove_code=True):
        """
        Plot delay/power vs code for direct CSV files (X, Y columns) with improved styling.
        Code is the row index (0-based).
        Generates both sweep and linearity (DNL/INL) plots with publication-ready styling.
        
        Args:
            filename: CSV filename with X and Y columns
            y_col_idx: Which column to use as Y data (default: 1 for second column)
            remove_code: If True, remove code at len(df)//2 + 1
        """
        df, path = self.load_data(filename)
        if df is None: 
            return
        
        # Load and preprocess data
        y_data = pd.to_numeric(df.iloc[:, y_col_idx], errors='coerce').dropna()
        original_length = len(y_data)        
        codes_to_remove = original_length // 2 
        print(len(y_data), codes_to_remove)
        
        # Remove middle code and re-index
        if remove_code and codes_to_remove < original_length:
            mask = np.arange(original_length) != codes_to_remove
            y_data = y_data[mask].reset_index(drop=True)
        
        codes = np.arange(len(y_data))
        metric_info = self._get_metric_info(path.stem)
        y_to_plot = (y_data.values - y_data.values[0]) * metric_info['scale']
        
        # Create sparse tick labels
        num_codes = len(codes)
        step = max(1, num_codes // 10)
        tick_positions = np.arange(0, num_codes, step)
        tick_labels = [str(int(c)) for c in codes[tick_positions]]
        
        # ===== PLOT 1: Sweep =====
        fig1, ax1 = self._create_figure(figsize=(11, 6))
        ax1.plot(codes, y_to_plot, marker=None, color=self.colors['primary'], 
             linewidth=2.6, label='Measured', alpha=0.9)
        
        thesis_title_sweep = self._get_thesis_title(filename, "sweep")
        clean_title = self._format_title(thesis_title_sweep)
        self._format_plot_labels(ax1, 
                                xlabel='Digital Code',
                                ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
                                title=clean_title
        )
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        ax1.legend(fontsize=11, framealpha=0.95)
        self._apply_grid_styling(ax1, alpha=0.35)
        
        plt.tight_layout()
        save_path1 = self._get_save_path(filename, metric_info['name'], "sweep_decimal")
        self._save_figure(fig1, save_path1)
        
        # ===== PLOT 2: DNL and INL =====
        y = y_data.values
        dnl, inl, lsb_ideal = self._compute_dnl_inl(y)
        if len(dnl) == 0:
            return
        
        fig2, (ax2, ax3) = self._create_figure(nrows=2, ncols=1, figsize=(11, 9), sharex=True)
        
        # DNL plot
        ax2.plot(codes, dnl, marker=None, color=self.colors['secondary'], 
            linewidth=2.6, label='DNL', alpha=0.9)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax2.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        thesis_title_dnl = self._get_thesis_title(filename, "linearity")
        clean_dnl_title = self._format_title(thesis_title_dnl)
        self._format_plot_labels(ax2, 
                                ylabel='DNL (LSB)',
                                title=clean_dnl_title
        )
        ax2.legend(fontsize=11, framealpha=0.95)
        self._apply_grid_styling(ax2, alpha=0.35)
        
        # INL plot
        ax3.plot(codes, inl, marker=None, color=self.colors['primary'], 
            linewidth=2.6, label='INL', alpha=0.9)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        self._format_plot_labels(ax3, 
                                xlabel='Digital Code',
                                ylabel='INL (LSB)',
                                title='Integral Non-Linearity'
        )
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(tick_labels)
        ax3.legend(fontsize=11, framealpha=0.95)
        self._apply_grid_styling(ax3, alpha=0.35)
        
        plt.tight_layout()
        save_path2 = self._get_save_path(filename, metric_info['name'], "linearity_decimal")
        self._save_figure(fig2, save_path2)


    def plot_histogram(self, filenames):
        """
        Plots histograms for Monte Carlo data with improved styling.
        If two files are provided, it calculates DR and Resolution per iteration.
        """
        # Handle single file or list of two files
        if isinstance(filenames, str):
            filenames = [filenames]
        
        datasets = []
        for f in filenames:
            df, path = self.load_data(f)
            if df is not None:
                datasets.append(df)
        
        if not datasets: return

        # --- Case A: Two Files (Calculate DR and Resolution) ---
        if len(datasets) == 2:
            # Assume col 0 is the data. Align indices to ensure iterations match.
            y1 = pd.to_numeric(datasets[0].iloc[:, 0], errors='coerce')
            y2 = pd.to_numeric(datasets[1].iloc[:, 0], errors='coerce')
            
            # DR = |File2 - File1|
            dr_data = (y2 - y1).abs().dropna()
            # Resolution = DR / (steps) -> assuming 5-bit (31 steps) if not specified
            res_data = dr_data / 31 
            
            metrics = {
                'Dynamic Range': dr_data,
                'Resolution': res_data
            }
            save_suffix = "mc_derived_metrics"
        
        # --- Case B: Single File (Standard Histogram) ---
        else:
            data = pd.to_numeric(datasets[0].iloc[:, 0], errors='coerce').dropna()
            metrics = {datasets[0].columns[0]: data}
            save_suffix = "mc_standard"

        # --- Plotting Loop with Enhanced Styling ---
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 7 * num_metrics))
        if num_metrics == 1: 
            axes = [axes]

        # Color palette for histograms
        histogram_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (ax, (label, data)) in enumerate(zip(axes, metrics.items())):
            # Create histogram with improved styling
            hist_color = histogram_colors[idx % len(histogram_colors)]
            counts, bins, patches = ax.hist(data, bins=40, color=hist_color, 
                                           edgecolor='black', linewidth=1.2, 
                                           alpha=0.75, label='Data Distribution')
            
            # Calculate statistics
            mean, std, median = data.mean(), data.std(), data.median()
            min_val, max_val = data.min(), data.max()
            
            # Plot vertical lines for statistics
            ax.axvline(mean, color='#d62728', linestyle='-', linewidth=2.5, 
                      label=f'Mean: {mean:.3e}', alpha=0.9)
            ax.axvline(median, color='#2ca02c', linestyle='--', linewidth=2, 
                      label=f'Median: {median:.3e}', alpha=0.8)
            ax.axvline(mean + std, color='#ff7f0e', linestyle=':', linewidth=2, 
                      label=f'Std Dev: {std:.3e}', alpha=0.8)
            ax.axvline(mean - std, color='#ff7f0e', linestyle=':', linewidth=2, alpha=0.8)
            
            # Highlight ±1σ region
            ax.axvspan(mean - std, mean + std, color='#ff7f0e', alpha=0.08)
            
            # Formatting
            thesis_title = self._get_thesis_title(filenames[0], "mc_distribution")
            clean_label = self._format_title(label)
            
            ax.set_title(thesis_title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel(clean_label, fontsize=12, fontweight='bold')
            ax.set_ylabel("Frequency (Count)", fontsize=12, fontweight='bold')
            
            # Add statistics box
            stats_text = f"Min: {min_val:.3e}\nMax: {max_val:.3e}\nRange: {max_val-min_val:.3e}"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            ax.set_axisbelow(True)

        plt.tight_layout()
        save_path = self._get_save_path(filenames[0], "MC_Analysis", save_suffix)
        self._save_figure(fig, save_path)

    def plot_corner_temperature_sweep(self, filename):
        """Plots metric vs temperature for different PVT corners."""
        df, path = self.load_data(filename)
        if df is None: return None
        x_cols = [c for c in df.columns if c.endswith(' X')]
        
        fig, ax = plt.subplots(figsize=(11, 6))
        corner_pattern = re.compile(r"top_(\w+)")
        for x_col in x_cols:
            y_col = x_col[:-2] + ' Y'
            if y_col in df.columns:
                match = corner_pattern.search(x_col)
                label = match.group(1).upper() if match else x_col
                ax.plot(df[x_col], df[y_col], marker=None, label=label, 
                       linewidth=2.6, alpha=0.9)
        
        ax.set_xlabel(r"Temperature [$^{\circ}$C]", fontsize=12, fontweight='bold')
        ylabel = self._format_title(path.stem.split('_')[0])
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        thesis_title = self._get_thesis_title(filename, "corner")
        ax.set_title(thesis_title, fontsize=14, fontweight='bold', pad=15)
        
        ax.legend(title="Corners", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"corner_T")
        self._save_figure(fig, save_path)

    def plot_corner_linearity_xy(self, filename):
        """Plot DNL/INL across corners from alternating X/Y columns with labels in row 0."""
        path = self.base_dir / filename
        if not path.exists():
            print(f"Warning: File {filename} not found.")
            return

        raw = pd.read_csv(path, header=0)
        if raw.shape[0] < 1 or raw.shape[1] < 2:
            print(f"Warning: File {filename} has insufficient data.")
            return

        labels = raw.columns.tolist()
        data = raw.reset_index(drop=True)

        corners = ["SS", "TT", "FF"]
        fig = {}
        ax_dnl = {}
        ax_inl = {}
        fig_sweep = {}
        ax_sweep = {}

        for corner in corners:
            fig[corner], (ax_dnl[corner], ax_inl[corner]) = plt.subplots(2, 1, figsize=(11, 10))
            fig[corner].patch.set_facecolor('white')
            fig[corner].patch.set_alpha(1.0)
            
            fig_sweep[corner], ax_sweep[corner] = self._create_figure(figsize=(11, 6))

        color_cycle = [
            self.colors['primary'], self.colors['secondary'], self.colors['tertiary'],
            self.colors['quaternary'], self.colors['accent'], self.colors['neutral']
        ]
        corner_color_idx = {corner: 0 for corner in corners}

        # Maps for Vdd-to-color and temperature-to-linestyle
        vdd_color_map = {}
        temp_linestyle_map = {
            -55: '-',      # solid line
            0: '--',       # dashed line
            27: ':',       # dotted line
        }

        def extract_params(label_str):
            """Extract corner, temperature, and Vdd from label string."""
            corner = None
            temp = None
            vdd = None

            label_lower = str(label_str).lower()

            # Extract corner from top_xx pattern (SS, TT, FF)
            corner_match = re.search(r'top_(ss|tt|ff)', label_lower)
            if corner_match:
                corner = corner_match.group(1).upper()

            # Extract temperature (temperature=-55 or T=-55)
            temp_match = re.search(r'temperature=(-?\d+(?:\.\d+)?)', label_lower)
            if temp_match:
                temp = int(float(temp_match.group(1)))

            # Extract Vdd (Vdd=0.88)
            vdd_match = re.search(r'vdd=(\d+(?:\.\d+)?)', label_lower)
            if vdd_match:
                vdd = float(vdd_match.group(1))

            return corner, temp, vdd

        for col in range(0, data.shape[1] - 1, 2):
            x = pd.to_numeric(data.iloc[:, col], errors='coerce')
            y = pd.to_numeric(data.iloc[:, col + 1], errors='coerce')
            mask = x.notna() & y.notna()
            if not mask.any():
                continue

            x_vals = x[mask].values
            y_vals = y[mask].values
            if len(y_vals) < 2:
                continue

            # Remove element 128 (index 127)
            if len(y_vals) > 128:
                x_vals = np.delete(x_vals, 128)
                y_vals = np.delete(y_vals, 128)

            label = labels[col + 1] if col + 1 < len(labels) else ""
            corner, temp, vdd = extract_params(label)
            if corner is None:
                continue

            # Assign color based on Vdd (same color for same voltage across all curves)
            if vdd not in vdd_color_map:
                vdd_color_map[vdd] = color_cycle[len(vdd_color_map) % len(color_cycle)]
            color = vdd_color_map[vdd]

            # Get linestyle based on temperature
            linestyle = temp_linestyle_map.get(temp, '-')

            # Build legend label
            legend_parts = []
            if temp is not None:
                legend_parts.append(f"T={temp}°C")
            if vdd is not None:
                legend_parts.append(f"Vdd={vdd}V")
            legend_label = ", ".join(legend_parts) if legend_parts else str(label)

            # Compute DNL and INL from delay data
            lsb_ideal = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)
            if lsb_ideal == 0 or np.isnan(lsb_ideal):
                continue
            
            dnl = np.diff(y_vals) / lsb_ideal - 1
            inl = np.cumsum(dnl)
            codes = np.arange(1, len(y_vals))

            # Plot DNL and INL on respective subplots
            ax_dnl[corner].plot(codes, dnl, label=legend_label, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9)
            ax_inl[corner].plot(codes, inl, label=legend_label, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.9)
            
            # Plot sweep (digital code vs output delay)
            ax_sweep[corner].plot(codes, y_vals[1:], label=legend_label, color=color, linestyle=linestyle, linewidth=2.4, alpha=0.9)

        thesis_title = self._get_thesis_title(filename, "linearity")
        clean_title = self._format_title(thesis_title)

        for corner in corners:
            if len(ax_dnl[corner].lines) == 0:
                continue

            # Format DNL subplot (top)
            ax_dnl[corner].set_title(f"{clean_title} - DNL ({corner})", fontsize=14, fontweight='bold')
            ax_dnl[corner].set_xlabel("Digital Code", fontsize=12, fontweight='bold')
            ax_dnl[corner].set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
            self._apply_grid_styling(ax_dnl[corner], alpha=0.35)
            ax_dnl[corner].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
            ax_dnl[corner].set_ylim(-1, 1)
            ax_dnl[corner].legend(fontsize=9, framealpha=0.95, loc='best')

            # Format INL subplot (bottom)
            ax_inl[corner].set_title(f"{clean_title} - INL ({corner})", fontsize=14, fontweight='bold')
            ax_inl[corner].set_xlabel("Digital Code", fontsize=12, fontweight='bold')
            ax_inl[corner].set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
            self._apply_grid_styling(ax_inl[corner], alpha=0.35)
            ax_inl[corner].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
            ax_inl[corner].legend(fontsize=9, framealpha=0.95, loc='best')

            plt.tight_layout()
            save_path = self._get_save_path(filename, "Linearity", f"corner_dnl_inl_{corner.lower()}")
            self._save_figure(fig[corner], save_path)
            
            # Format and save sweep plot
            ax_sweep[corner].set_title(f"Sweep Plot - Code vs Output Delay ({corner})", fontsize=14, fontweight='bold')
            ax_sweep[corner].set_xlabel("Digital Code", fontsize=12, fontweight='bold')
            ax_sweep[corner].set_ylabel("Output Delay (s)", fontsize=12, fontweight='bold')
            self._apply_grid_styling(ax_sweep[corner], alpha=0.35)
            ax_sweep[corner].legend(fontsize=9, framealpha=0.95, loc='best')
            
            plt.tight_layout()
            save_path_sweep = self._get_save_path(filename, "Linearity", f"corner_sweep_{corner.lower()}")
            self._save_figure(fig_sweep[corner], save_path_sweep)

    def plot_generic_sweep(self, filename):
        """Format: Single X and Y pair."""
        df, path = self.load_data(filename)
        x_label = self._get_axis_labels(filename) # <--- Dynamically gets 'Vdd', 'Cap', etc.

        if df is None: return None
        x_cols = [c for c in df.columns if c.endswith(' X')]
        y_cols = [c for c in df.columns if c.endswith(' Y')]
        if not x_cols or not y_cols: return None
        
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(df[x_cols[0]], df[y_cols[0]], marker=None, color=self.colors['primary'], 
               linewidth=2.6, label=y_cols[0].replace(' Y', ''))
        
        title = self._format_title(path.stem)
        ax.set_title(f"Sweep: {title}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_cols[0].replace(' Y', ''), fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"generic_sweep")
        self._save_figure(fig, save_path)
    
        plt.close()

    def plot_signals(self, filename, filters=None, t_range=None, subplots=False):
        """
        'filters': List of strings (OR logic for selection).
        't_range': Tuple (t_start, t_stop).
        'subplots': If True, splits signals into different axes based on signal name.
        """
        df, path = self.load_data(filename)
        if df is None: return None        

        # 1. Filter columns
        x_cols = [c for c in df.columns if c.endswith(' X')]
        if filters:
            if isinstance(filters, str): filters = [filters]
            x_cols = [c for c in x_cols if any(f.lower() in c.lower() for f in filters)]

        if not x_cols:
            print("No matching signals found.")
            return None

        # 2. Group signals by their base name (the part before bit parameters)
        groups = {}
        for xc in x_cols:
            base_name = xc[:-2].lstrip('/').split(' (')[0]
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(xc)

        unique_signals = list(groups.keys())
        num_signals = len(unique_signals)

        # 3. Create Figure and Axes
        if subplots and num_signals > 1:
            fig, axes = plt.subplots(num_signals, 1, sharex=True, figsize=(12, 4 * num_signals))
            if num_signals == 1: axes = [axes]
        else:
            fig = plt.figure(figsize=(12, 6))
            axes = [plt.gca()] * num_signals # All point to same axis if subplots=False

        # 4. Plot each group
        for idx, base_name in enumerate(unique_signals):
            ax = axes[idx]
            for x_col in groups[base_name]:
                y_col = x_col[:-2] + ' Y'
                x_vals = pd.to_numeric(df[x_col], errors='coerce')
                y_vals = pd.to_numeric(df[y_col], errors='coerce')
                
                mask = x_vals.notna() & y_vals.notna()
                if t_range:
                    mask = mask & (x_vals >= t_range[0]) & (x_vals <= t_range[1])
                
                if not any(mask): continue
                
                # Use only the bit part (e.g. "(d0=0,...)") for the legend to save space
                label_match = re.search(r"(\(.*\))", x_col)
                label = label_match.group(1) if label_match else self._format_title(base_name)

                ax.plot(x_vals[mask], y_vals[mask], label=label, linewidth=2, markersize=4, marker='o')
            
            clean_signal_name = self._format_title(base_name)
            ax.set_ylabel("Value", fontsize=11, fontweight='bold')
            ax.set_title(f"Signal: {clean_signal_name}", fontsize=13, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
            if t_range:
                ax.set_xlim(t_range)

        plt.xlabel("Time (s)", fontsize=11, fontweight='bold')
        plt.tight_layout()

        save_path = self._get_save_path(filename, self._format_title(unique_signals[0]) ,"transient_signals")
        self._save_figure(fig, save_path)
        
        plt.close()

    def plot_pvt_sweep(self, filename, subfigure=False, VDD=False):
        df, path = self.load_data(filename)
        if df is None: return

        # 1. Standard Cleaning & Column Detection
        df['delay'] = pd.to_numeric(df['delay'], errors='coerce')
        df['P_avg'] = pd.to_numeric(df['P_avg'], errors='coerce')
        
        vdd_col = None
        if VDD:
            vdd_col = next((c for c in df.columns if c.upper() == 'VDD'), None)
            if vdd_col:
                df[vdd_col] = pd.to_numeric(df[vdd_col], errors='coerce')
            else:
                VDD = False

        df = df.dropna(subset=['delay'])
        df['code'] = df['d0'] + 2*df['d1'] + 4*df['d2'] + 8*df['d3'] + 16*(1 - df['d4'])
        df = df[df['code'] != 16]

        # 2. Base Style Dictionaries
        df['base_corner'] = df['Corner'].str.split('_').str[0].str.upper()
        
        # --- Mode A: VDD is False (Color = Corner, Style = Temp) ---
        corner_colors_std = {'SS': '#d62728', 'TT': '#1f77b4', 'FF': '#2ca02c', 'SF': '#ff7f0e', 'FS': '#9467bd'}
        temp_styles_std = {80: '-', 0: '--', -55: ':'} 

        # --- Mode B: VDD is True (Color = Corner_Temp, Style = VDD) ---
        pvt_group_colors = {
            'SS_80': '#8b0000', 'SS_0': '#d62728', 'SS_-55': '#ff6b6b',
            'TT_80': '#08519c', 'TT_0': '#3182bd', 'TT_-55': '#9ecae1',
            'FF_80': '#006d2c', 'FF_0': '#31a354', 'FF_-55': '#74c476'
        }
        vdd_styles = ['-', '--', ':', '-.']

        def plot_metric(y_col, y_label, scale, unit, suffix):
            unique_corners = sorted(df['base_corner'].unique())
            
            if subfigure:
                fig, axes = plt.subplots(len(unique_corners), 1, figsize=(14, 6 * len(unique_corners)), sharex=True)
                if len(unique_corners) == 1: axes = [axes]
            else:
                fig = plt.figure(figsize=(14, 8))
                axes = [plt.gca()] * len(unique_corners)

            group_cols = ['Corner', 'temperature', vdd_col] if VDD else ['Corner', 'temperature']
            
            for i, base_corner in enumerate(unique_corners):
                ax = axes[i]
                corner_group = df[df['base_corner'] == base_corner]
                
                if VDD:
                    unique_vdds = sorted(corner_group[vdd_col].unique())
                    v_style_map = {v: vdd_styles[idx % len(vdd_styles)] for idx, v in enumerate(unique_vdds)}

                for names, group in corner_group.groupby(group_cols):
                    c_full, temp = names[0], names[1]
                    v_val = names[2] if VDD else None
                    
                    group = group.sort_values('code')
                    x_labels = group['code'].apply(lambda c: bin(c)[2:].zfill(5))
                    y_vals = group[y_col].values * scale
                    dr = y_vals.max() - y_vals.min()
                    
                    lbl = f"{c_full}, {temp}°C"
                    if VDD: lbl += f", {v_val}V"
                    lbl += f" | DR: {dr:.2f}{unit}"

                    # --- STYLING LOGIC ---
                    if VDD:
                        color_key = f"{base_corner}_{temp}"
                        line_color = pvt_group_colors.get(color_key, 'black')
                        line_style = v_style_map[v_val]
                    else:
                        line_color = corner_colors_std.get(base_corner, 'black')
                        line_style = temp_styles_std.get(temp, '-')

                    ax.plot(x_labels, y_vals, label=lbl, color=line_color, 
                            linestyle=line_style, marker='o', markersize=4, alpha=0.85, linewidth=2)

                corner_label = self._format_title(f"{base_corner}")
                if subfigure:
                    ax.set_title(f"Corner: {corner_label}", fontsize=12, fontweight='bold', pad=10)
                ax.set_ylabel(f"{y_label} ({unit})", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, framealpha=0.95)
                ax.set_axisbelow(True)

            if not subfigure:
                axes[0].set_title(f"{y_label} vs Digital Code", fontsize=14, fontweight='bold', pad=15)
            
            axes[-1].set_xlabel("Digital Code ($b_4b_3b_2b_1b_0$)", fontsize=11, fontweight='bold')
            axes[-1].tick_params(axis='x', rotation=45, labelsize=9)
            plt.tight_layout()
            
            final_suffix = f"{suffix}_subplots" if subfigure else suffix
            save_path = self._get_save_path(filename, y_label.replace(' ', '_'), final_suffix)
            self._save_figure(fig, save_path)

        plot_metric('delay', 'Delay', 1e9, 'ns', 'pvt_delay')
        plot_metric('P_avg', 'Average Power', 1e6, 'µW', 'pvt_power')

    def plot_pvt_linearity(self, filename, subplots=False):
        """
        Calculates DNL and INL for PVT data.
        - subplots: If True, creates a separate row for each corner.
        """
        df, path = self.load_data(filename)
        if df is None: return

        # 1. Clean and Reconstruct
        df['delay'] = pd.to_numeric(df['delay'], errors='coerce')
        df = df.dropna(subset=['delay'])
        df['code'] = df['d0'] + 2*df['d1'] + 4*df['d2'] + 8*df['d3'] + 16*(1 - df['d4'])
        df = df[df['code'] != 16]

        # 2. Setup Styles
        df['base_corner'] = df['Corner'].str.split('_').str[0].str.upper()
        unique_corners = sorted(df['base_corner'].unique())
        corner_colors = {'SS': '#d62728', 'TT': '#1f77b4', 'FF': '#2ca02c', 'SF': '#ff7f0e', 'FS': '#9467bd'}
        temp_styles = {80: '-', 0: '--', -55: ':'} 

        # 3. Dynamic Figure Setup
        if subplots:
            num_rows = len(unique_corners)
            fig, axes = plt.subplots(num_rows, 2, figsize=(16, 5 * num_rows), sharex=True)
            # Ensure axes is 2D even if only one corner exists
            if num_rows == 1: axes = np.expand_dims(axes, axis=0)
        else:
            fig, (ax_dnl, ax_inl) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            # Duplicate the axes references so the loop logic remains the same
            axes = [[ax_dnl, ax_inl]] * len(unique_corners)

        # 4. Process Each Corner
        for i, base_corner in enumerate(unique_corners):
            cur_ax_dnl = axes[i][0]
            cur_ax_inl = axes[i][1]
            
            corner_group = df[df['base_corner'] == base_corner]
            
            for (corner_full, temp), group in corner_group.groupby(['Corner', 'temperature']):
                group = group.sort_values('code')
                y = group['delay'].values
                x_labels = group['code'].apply(lambda c: bin(c)[2:].zfill(5))

                dnl, inl, _ = self._compute_dnl_inl(y)
                if len(dnl) == 0:
                    continue

                color = corner_colors.get(base_corner, 'black')
                style = temp_styles.get(temp, '-')
                lbl = f"{corner_full}, {temp}°C"

                cur_ax_dnl.plot(x_labels, dnl, label=lbl, color=color, linestyle=style, 
                               marker='o', markersize=3, alpha=0.7, linewidth=2)
                cur_ax_inl.plot(x_labels, inl, label=lbl, color=color, linestyle=style, 
                               marker='o', markersize=3, alpha=0.7, linewidth=2)

            # Labels for Subplot Mode
            if subplots:
                corner_label = self._format_title(f"{base_corner}")
                cur_ax_dnl.set_title(f"DNL - {corner_label}", fontsize=12, fontweight='bold', pad=10)
                cur_ax_inl.set_title(f"INL - {corner_label}", fontsize=12, fontweight='bold', pad=10)
                cur_ax_inl.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.95)
            
            # Add reference lines
            cur_ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
            cur_ax_dnl.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            cur_ax_dnl.axhline(y=-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            cur_ax_inl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
            
            cur_ax_dnl.grid(True, alpha=0.3)
            cur_ax_inl.grid(True, alpha=0.3)
            cur_ax_dnl.set_axisbelow(True)
            cur_ax_inl.set_axisbelow(True)

        # 5. Global Formatting
        if not subplots:
            axes[0][0].set_title("Differential Non-Linearity (DNL)", fontsize=14, fontweight='bold', pad=15)
            axes[0][1].set_title("Integral Non-Linearity (INL)", fontsize=14, fontweight='bold', pad=15)
            axes[0][0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, framealpha=0.95)
            
            axes[0][0].set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
            axes[0][1].set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')

        axes[-1][0].set_xlabel("Digital Code ($b_4b_3b_2b_1b_0$)", fontsize=11, fontweight='bold')
        axes[-1][0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[-1][1].tick_params(axis='x', rotation=45, labelsize=9)

        plt.tight_layout()
        suffix = "pvt_linearity_split" if subplots else "pvt_linearity_combined"
        save_path = self._get_save_path(filename, "Linearity", suffix)
        self._save_figure(fig, save_path)

    def available_plot_types(self):
        """Return supported plot types for CLI and scripting usage."""
        return {
            "auto": "Auto-detect best plotting method",
            "sweep": "Digital/code sweep",
            "linearity": "DNL/INL linearity",
            "signals": "Transient signals",
            "histogram": "Histogram / MC distribution",
            "mc_sweep": "MC sweep all realizations",
            "mc_linearity": "MC linearity all realizations",
            "mc_linearity_iteration": "MC linearity per iteration",
            "mcparamset_sweep": "MC paramset sweeps",
            "mcparamset_linearity": "MC paramset DNL/INL",
            "direct_csv": "Direct CSV sweep + linearity",
            "avg_power": "Average power by code",
            "corner_temp": "Corner temperature sweep",
            "corner_linearity": "Corner DNL/INL",
            "generic": "Generic single X/Y sweep",
            "pvt_sweep": "PVT sweep",
            "pvt_linearity": "PVT linearity",
        }

    def plot_file(self, filename, plot_type="auto", **kwargs):
        """Single entry point for plotting a file by explicit plot type."""
        plot_type = (plot_type or "auto").lower().strip()

        if plot_type == "linearity":
            df_preview, _ = self.load_data(filename)
            if df_preview is not None:
                has_bit_cols = any(c.startswith('d') and c[1:].isdigit() for c in df_preview.columns)
                has_xy_pair = len(df_preview.columns) >= 2 and str(df_preview.columns[0]).endswith(' X') and str(df_preview.columns[1]).endswith(' Y')
                if has_xy_pair and not has_bit_cols:
                    print("Info: Detected direct X/Y CSV. Using direct_csv mode for linearity and sweep.")
                    return self.plot_direct_csv_sweep(
                        filename,
                        y_col_idx=kwargs.get("y_col_idx", 1),
                        remove_code=kwargs.get("remove_code", True),
                    )

        dispatch = {
            "auto": lambda: self.smart_plot(filename),
            "sweep": lambda: self.plot_digital_sweep(filename, max_iterations=kwargs.get("max_iterations", 200)),
            "linearity": lambda: self.plot_linearity(filename),
            "signals": lambda: self.plot_signals(
                filename,
                filters=kwargs.get("filters"),
                t_range=kwargs.get("t_range"),
                subplots=kwargs.get("subplots", False),
            ),
            "histogram": lambda: self.plot_histogram(filename),
            "mc_sweep": lambda: self.plot_mc_sweep_all_realizations(filename, max_realizations=kwargs.get("max_realizations", 200)),
            "mc_linearity": lambda: self.plot_mc_linearity_all_realizations_xy(filename, max_realizations=kwargs.get("max_realizations", 200)),
            "mc_linearity_iteration": lambda: self.plot_mc_linearity_per_iteration(
                filename,
                lsb_ns=kwargs.get("lsb_ns"),
                max_iterations=kwargs.get("max_iterations", 200),
            ),
            "mcparamset_sweep": lambda: self.plot_mcparamset_sweep(filename, max_realizations=kwargs.get("max_realizations", 200)),
            "mcparamset_linearity": lambda: self.plot_mcparamset_linearity(filename, max_realizations=kwargs.get("max_realizations", 200)),
            "direct_csv": lambda: self.plot_direct_csv_sweep(
                filename,
                y_col_idx=kwargs.get("y_col_idx", 1),
                remove_code=kwargs.get("remove_code", True),
            ),
            "avg_power": lambda: self.plot_average_power_by_code(
                filename,
                start_time=kwargs.get("start_time", 2e-8),
                window_size=kwargs.get("window_size", 5e-8),
                num_codes=kwargs.get("num_codes", 256),
                remove_code=kwargs.get("remove_code", True),
                P_static=kwargs.get("P_static", 0),
            ),
            "corner_temp": lambda: self.plot_corner_temperature_sweep(filename),
            "corner_linearity": lambda: self.plot_corner_linearity_xy(filename),
            "generic": lambda: self.plot_generic_sweep(filename),
            "pvt_sweep": lambda: self.plot_pvt_sweep(filename, subfigure=kwargs.get("subfigure", False), VDD=kwargs.get("vdd", False)),
            "pvt_linearity": lambda: self.plot_pvt_linearity(filename, subplots=kwargs.get("subplots", False)),
        }

        if plot_type not in dispatch:
            valid = ", ".join(sorted(dispatch.keys()))
            raise ValueError(f"Unknown plot type '{plot_type}'. Valid options: {valid}")

        return dispatch[plot_type]()

    def smart_plot(self, filename):
        """Routes any filename to the correct plotting function."""
        df, path = self.load_data(filename)
        if df is None: return
        
        name = filename.lower()
        
        # 1. Monte Carlo
        if "mc" in name:
            return self.plot_histogram(filename)
        
        # 2. Corner/Temp Sweeps
        if any(k in name for k in ["corner"]):
            return self.plot_corner_temperature_sweep(filename)
        
        # 3. Digital Code Sweeps (detect presence of d0, d1 or 'code')
        if any(k in name for k in ["code", "bit"]):
            # If the user specifically wants linearity for certain files:
            if "linearity" in name or "constant_slope" in name:
                return self.plot_linearity(filename)
            return self.plot_digital_sweep(filename)
            
        # 4. Standard Signal vs Time
        x_cols = [c for c in df.columns if c.endswith(' X')]
        if len(x_cols) > 1:
            return self.plot_signals(filename, subplots=True)
        
        return self.plot_generic_sweep(filename)


# ============================================================================
# PLOT CONFIGURATION & EXECUTION
# ============================================================================

def define_plot_tasks():
    """Define all plotting tasks in a structured format."""
    return {
        "constant_slope_sweeps": {
            "description": "Constant slope automatic sweeps",
            "files": [
                "cs_delay_code_4bit.csv", "cs_delay_code_5bit.csv", "cs_delay_code_8bit.csv",
                "cs_delay_code_5bit_nonidealcurs.csv", "cs_delay_code_5bit_coarse.csv",
                "cs_power_code_4bit.csv", "cs_power_code_5bit.csv", "cs_delay_code_6bit",
                "cs_power_code_5bit_nonidealcurs.csv", "cs_power_code_5bit_coarse.csv",
                "cs_delay_mc_tt_00000.csv", "cs_power_mc_tt_00000.csv", 
                "cs_delay_mc_tt_11111.csv", "cs_power_mc_tt_11111.csv",  
                "cs_delay_mc_tt_10000.csv", "cs_power_mc_tt_10000.csv",
            ],
            "action": lambda plotter, f: plotter.smart_plot(f)
        },
        "constant_slope_direct_csv": {
            "description": "Constant slope direct CSV sweeps with decimal codes",
            "files": ["cs_delay_code_5bit_counter.csv"],
            "action": lambda plotter, f: plotter.plot_direct_csv_sweep(f, y_col_idx=1, remove_code=True)
        },
        "constant_slope_linearity": {
            "description": "Constant slope linearity (DNL/INL) analysis",
            "files": [
                "cs_delay_code_4bit.csv", "cs_delay_code_5bit.csv", "cs_delay_code_8bit.csv",
                "cs_delay_code_5bit_coarse.csv", "cs_delay_code_5bit_nonidealcurs.csv",
                "cs_delay_code_5bit_counter.csv", "cs_delay_code_6bit.csv"
            ],
            "action": lambda plotter, f: plotter.plot_linearity(f)
        },
        "constant_slope_mc_linearity": {
            "description": "Constant slope MC linearity per iteration",
            "files": ["cs_delay_code_8bit_counter_mc.csv", "cs_delay_code_8bit_mc.csv"],
            "action": lambda plotter, f: plotter.plot_mc_linearity_all_realizations_xy(f, max_realizations=200)
        },
        "constant_slope_transients_4bit": {
            "description": "Constant slope 4-bit transient signals",
            "files": ["cs_Pinst_code_4bit.csv", "cs_signals_code_4bit.csv"],
            "configs": [
                {
                    "file": "cs_Pinst_code_4bit.csv",
                    "filters": ["d0=0,d1=0,d2=0,d3=0", "d0=0,d1=0,d2=0,d3=1", 
                                "d0=1,d1=1,d2=1,d3=0", "d0=1,d1=1,d2=1,d3=1"],
                    "t_range": (40e-9, 100e-9)
                },
                {
                    "file": "cs_signals_code_4bit.csv",
                    "filters": ["d0=0,d1=0,d2=0,d3=0", "d0=0,d1=0,d2=0,d3=1", 
                                "d0=1,d1=1,d2=1,d3=0", "d0=1,d1=1,d2=1,d3=1"],
                    "t_range": (40e-9, 100e-9)
                }
            ],
            "action": lambda plotter, f, config: plotter.plot_signals(config["file"], filters=config["filters"], t_range=config["t_range"], subplots=True)
        },
        "constant_slope_transients_5bit": {
            "description": "Constant slope 5-bit transient signals",
            "files": ["cs_Pinst_code_5bit.csv", "cs_signals_code_5bit.csv"],
            "configs": [
                {
                    "file": "cs_Pinst_code_5bit.csv",
                    "filters": ["d0=0,d1=0,d2=0,d3=0,d4=0", "d0=0,d1=0,d2=0,d3=1,d4=1", 
                                "d0=0,d1=0,d2=0,d3=0,d4=1", "d0=1,d1=1,d2=1,d3=0,d4=0", 
                                "d0=1,d1=1,d2=1,d3=1,d4=0"],
                    "t_range": (40e-9, 120e-9)
                },
                {
                    "file": "cs_signals_code_5bit.csv",
                    "filters": ["d0=0,d1=0,d2=0,d3=0,d4=0", "d0=0,d1=0,d2=0,d3=1,d4=1", 
                                "d0=0,d1=0,d2=0,d3=0,d4=1", "d0=1,d1=1,d2=1,d3=0,d4=0", 
                                "d0=1,d1=1,d2=1,d3=1,d4=0"],
                    "t_range": (40e-9, 120e-9)
                }
            ],
            "action": lambda plotter, f, config: plotter.plot_signals(config["file"], filters=config["filters"], t_range=config["t_range"], subplots=True)
        },
        "constant_slope_pvt": {
            "description": "Constant slope PVT analysis",
            "files": ["cs_delay_power_corner_T.csv", "cs_delay_power_VDD_corner_T.csv"],
            "custom": True  # Handled specially
        },
        "constant_slope_corner_linearity": {
            "description": "Constant slope corner DNL/INL and sweep",
            "files": ["cs_delay_code_8bit_corner.csv"],
            "action": lambda plotter, f: plotter.plot_corner_linearity_xy(f)
        },
        "constant_slope_digital_sweep_mc": {
            "description": "Constant slope digital sweep MC all iterations",
            "files": ["cs_delay_code_8bit_counter_mc.csv", "cs_delay_code_8bit_mc.csv"],
            "action": lambda plotter, f: plotter.plot_mc_sweep_all_realizations(f, max_realizations=200)
        },
        "constant_slope_mcparamset": {
            "description": "Constant slope mcparamset sweep and linearity",
            "files": ["cs_delay_8bit_mc_idcursrc.csv"],
            "action": lambda plotter, f: (plotter.plot_mcparamset_sweep(f, max_realizations=200),
                                           plotter.plot_mcparamset_linearity(f, max_realizations=200))
        },
        "constant_slope_avg_power": {
            "description": "Constant slope average power analysis by code",
            "files": [("cs_Pinst_code_8bit.csv",8, 5e-8), ("cs_power_code_6bit.csv",6,2e-8)],
            "action": lambda plotter, f: plotter.plot_average_power_by_code(f[0], num_codes=2**f[1]-1, window_size=f[2])
        },
        "constant_slope_histogram": {
            "description": "Constant slope DR/Resolution histogram",
            "files": ["cs_delay_mc_tt_00000.csv", "cs_delay_mc_tt_11111.csv"],
            "custom": True  # Handled specially
        },
        "variable_slope_sweeps": {
            "description": "Variable slope automatic sweeps",
            "files": ["dlcsi_delay_code_5bit.csv", "dlcsi_power_code_5bit.csv"],
            "action": lambda plotter, f: plotter.smart_plot(f)
        },
        "variable_slope_linearity": {
            "description": "Variable slope linearity analysis",
            "files": ["dlcsi_delay_code_5bit.csv"],
            "action": lambda plotter, f: plotter.plot_linearity(f)
        },
        "phase_interpolator_sweeps": {
            "description": "Phase interpolator automatic sweeps",
            "files": ["pi_delay_code_4bit.csv", "pi_power_code_4bit.csv"],
            "action": lambda plotter, f: plotter.smart_plot(f)
        },
        "phase_interpolator_linearity": {
            "description": "Phase interpolator linearity analysis",
            "files": ["pi_delay_code_4bit.csv"],
            "action": lambda plotter, f: plotter.plot_linearity(f)
        }
    }


def run_plots(plotter, plot_all=True, selected_tasks=None):
    """
    Run plotting tasks with flexible options.
    
    Args:
        plotter: CadencePlotter instance
        plot_all: If True, run all tasks. If False, use selected_tasks
        selected_tasks: List of task names to run (e.g., ["constant_slope_sweeps", "constant_slope_linearity"])
    """
    tasks = define_plot_tasks()
    
    if plot_all:
        tasks_to_run = tasks.items()
    elif selected_tasks:
        tasks_to_run = [(name, tasks[name]) for name in selected_tasks if name in tasks]
    else:
        print("No tasks selected. Use plot_all=True or specify selected_tasks.")
        return
    
    for task_name, task_config in tasks_to_run:
        print(f"\n{'='*70}")
        print(f"Running: {task_name}")
        print(f"Description: {task_config['description']}")
        print(f"{'='*70}")
        
        try:
            if task_config.get("custom"):
                # Handle custom tasks
                if task_name == "constant_slope_pvt":
                    plotter.plot_pvt_sweep("cs_delay_power_corner_T.csv", subfigure=False)
                    plotter.plot_pvt_sweep("cs_delay_power_VDD_corner_T.csv", VDD=True, subfigure=True)
                    plotter.plot_pvt_linearity("cs_delay_power_corner_T.csv", subplots=False)
                elif task_name == "constant_slope_histogram":
                    plotter.plot_histogram(["cs_delay_mc_tt_00000.csv", "cs_delay_mc_tt_11111.csv"])
            
            elif "configs" in task_config:
                # Handle tasks with custom configurations
                for config in task_config["configs"]:
                    plotter.plot_signals(config["file"], filters=config["filters"], 
                                       t_range=config["t_range"], subplots=True)
            
            else:
                # Handle standard tasks
                action = task_config["action"]
                for file in task_config["files"]:
                    try:
                        action(plotter, file)
                    except Exception as e:
                        print(f"  Error processing {file}: {str(e)}")
            
            print(f"[OK] {task_name} completed")
        
        except Exception as e:
            print(f"[ERROR] {task_name} failed: {str(e)}")


def _parse_t_range(value):
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("t-range must be in form 'start,stop'")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("t-range values must be numeric") from exc


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Flexible Cadence CSV plotting utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=["jhelp"],
                        help="Quick help command (run: python plot_delay.py jhelp)")
    parser.add_argument("--base-dir", default="results_cadence", help="Directory containing CSV files")
    parser.add_argument("--plot-dir", default="C:\\Users\\zipar\\OneDrive - Delft University of Technology\\Second Year\\MEP\\plots", help="Directory where plots are saved")
    parser.add_argument("--file", help="CSV filename to plot")
    parser.add_argument("--type", default="auto", help="Plot type (use --list-types to see options)")
    parser.add_argument("--list-types", action="store_true", help="List supported plot types and exit")
    parser.add_argument("--task", action="append", help="Run a predefined task name (can repeat)")
    parser.add_argument("--all-tasks", action="store_true", help="Run all predefined tasks")
    parser.add_argument("--coarse-fine", action="store_true", help="Run coarse-fine processing flow")
    parser.add_argument("--P-static", type=float, default=0, help="Static power to subtract (default: 0)")
    parser.add_argument("--max-realizations", type=int, default=200)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--lsb-ns", type=float, default=None)
    parser.add_argument("--y-col-idx", type=int, default=1)
    parser.add_argument("--no-remove-code", action="store_true")
    parser.add_argument("--num-codes", type=int, default=256)
    parser.add_argument("--window-size", type=float, default=5e-8)
    parser.add_argument("--start-time", type=float, default=2e-8)
    parser.add_argument("--subplots", action="store_true")
    parser.add_argument("--subfigure", action="store_true")
    parser.add_argument("--vdd", action="store_true")
    parser.add_argument("--filters", nargs='*', default=None, help="Signal filters (for signals plot)")
    parser.add_argument("--t-range", type=_parse_t_range, default=None, help="Time range: start,stop")

    return parser


def print_jhelp(plotter):
    print("\nJHELP - Cadence Plot Quick Guide")
    print("=" * 60)
    print("\n1) Most common commands")
    print("  python plot_delay.py --list-types")
    print("  python plot_delay.py --file your_file.csv --type auto")
    print("  python plot_delay.py --file cs_delay_code_4bit.csv --type linearity")
    print("  python plot_delay.py --file cs_delay_code_8bit_counter_mc.csv --type mc_linearity --max-realizations 200")
    print("\n2) Important options")
    print("  --base-dir results_cadence   Where CSV files are")
    print("  --plot-dir plots             Where output images go")
    print("  --type <plot_type>           Which plot style to generate")
    print("\n3) Available plot types")
    for name, desc in plotter.available_plot_types().items():
        print(f"  - {name:22s} {desc}")
    print("\nTip: If unsure, use --type auto.\n")


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    plotter = CadencePlotter(base_dir=args.base_dir, plot_dir=args.plot_dir)

    if args.command == "jhelp":
        print_jhelp(plotter)
        return

    if args.list_types:
        print("Available plot types:")
        for name, desc in plotter.available_plot_types().items():
            print(f"  - {name:22s} {desc}")
        return

    if args.all_tasks or args.task:
        run_plots(plotter, plot_all=args.all_tasks, selected_tasks=args.task)
        return

    if args.coarse_fine:
        from process_coarse_fine import process_coarse_fine
        process_coarse_fine()
        return

    if not args.file:
        parser.error("Provide --file for single-file plotting, or use --task / --all-tasks")

    plotter.plot_file(
        args.file,
        plot_type=args.type,
        P_static=args.P_static,
        max_realizations=args.max_realizations,
        max_iterations=args.max_iterations,
        lsb_ns=args.lsb_ns,
        y_col_idx=args.y_col_idx,
        remove_code=not args.no_remove_code,
        num_codes=args.num_codes,
        window_size=args.window_size,
        start_time=args.start_time,
        subplots=args.subplots,
        subfigure=args.subfigure,
        vdd=args.vdd,
        filters=args.filters,
        t_range=args.t_range,
    )


# --- Usage ---
if __name__ == "__main__":
    main()