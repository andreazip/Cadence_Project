import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import numpy as np
from pathlib import Path

# ============================================================================
# PUBLICATION-READY PLOT STYLE
# ============================================================================
# Set font for scientific publications - Arial is universally available
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

plt.rcParams.update({
    # Font sizes
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    
    # Line and marker styles
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.2,
    
    # Grid
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    
    # Figure
    "figure.dpi": 100,
    "savefig.dpi": 300,  # High resolution for saving
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    
    # Axes
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "xtick.major.width": 1.2,
    "xtick.minor.width": 0.8,
    "ytick.major.width": 1.2,
    "ytick.minor.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
})

class CadencePlotter:
    def __init__(self,base_dir="results_cadence",plot_dir=r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\\plots"):
        self.base_dir = Path(base_dir)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Color palette for consistent styling
        self.colors = {
            'primary': '#D62728',      # Crimson
            'secondary': '#1F77B4',    # Blue
            'tertiary': '#2CA02C',     # Green
            'quaternary': '#FF7F0E'    # Orange
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
    
    def _apply_grid_styling(self, ax, alpha=0.3):
        """Apply consistent grid styling."""
        ax.grid(True, alpha=alpha, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
    
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
        if not path.exists():
            path = self.base_dir / filename
        if not path.exists():
            print(f"Warning: File {filename} not found.")
            return None, path
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df, path

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

        if is_cs:
            if 15 in df_clean['code'].values:
                df_clean = df_clean[df_clean['code'] != 15].reset_index(drop=True)

        df_clean['label'] = df_clean['code'].apply(lambda c: bin(int(c))[2:].zfill(bit_count))
        df_clean['code_index'] = np.arange(len(df_clean))
        df_clean['label_index'] = df_clean['code_index'].apply(lambda c: bin(int(c))[2:].zfill(bit_count))
        df_clean['y'] = df_clean[metric_col]
        return df_clean

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
        
        # --- Remove Code 16 ONLY if it is a constant slope (cs) file ---
        if is_cs:
            if 15 in plot_df['code'].values:
                plot_df = plot_df[plot_df['code'] != 15].reset_index(drop=True)
        
        return plot_df

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

    def plot_digital_sweep(self, filename, signal_name=None, max_iterations=20):
        """Plot digital code sweeps with improved styling and metrics."""
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        if signal_name is None:
            signal_name = df.columns[0].split(' (')[0].split(' ')[0]

        # Check if this is MC code sweep format
        has_bit_cols = all(f'd{i}' in df.columns for i in range(bit_count))
        has_x_y_cols = any(c.endswith(' X') or c.endswith('X') for c in df.columns)
        
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
                y_to_plot = (y - y[0]) * metric_info['scale'] if not metric_info['is_power'] else y * metric_info['scale']
                
                alpha = max(0.1, 1.0 - idx/len(iterations))
                ax.plot(plot_df['label'], y_to_plot, marker='o', linewidth=1.5, 
                       alpha=alpha, color=self.colors['secondary'],
                       label=f"Run {idx+1}" if idx % 5 == 0 else None)
            
            self._format_plot_labels(ax,
                                    xlabel=f"Digital Code",
                                    ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
                                    title=f"{bit_count}-Bit Monte Carlo Sweep - {path.stem.replace('_', ' ').title()}"
            )
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=9, loc='best', ncol=2)
            self._apply_grid_styling(ax)
            
            plt.tight_layout()
            save_path = self._get_save_path(filename, signal_name, "sweep_mc_all_iterations")
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
        ax.plot(plot_df['label'], y_to_plot, marker='o', color=self.colors['primary'], 
                linewidth=2.5, markersize=5, 
                label=f"DR: {dr:.3f}{metric_info['prefix']}{metric_info['unit']} | "
                      f"Res: {res*1000:.2f}p{metric_info['unit']}")
        
        self._format_plot_labels(ax,
                                xlabel=f"Digital Code (LSB={bit_count-1}...0)",
                                ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
                                title=f"{bit_count}-Bit Digital Sweep"
        )
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=10, loc='best')
        self._apply_grid_styling(ax)

        plt.tight_layout()
        save_path = self._get_save_path(filename, signal_name, "sweep")
        self._save_figure(fig, save_path)

    def plot_mc_linearity_per_iteration(self, filename, lsb_ns=None, max_iterations=20):
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
        fig, (ax_dnl, ax_inl) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Iterate through each MC realization
        iterations = sorted(df['mc_iteration'].unique())[:max_iterations]
        num_iterations = len(iterations)
        
        # Create a colormap for iterations
        colors = plt.cm.viridis(np.linspace(0, 1, num_iterations))
        
        for idx, iteration in enumerate(iterations):
            df_iter = df[df['mc_iteration'] == iteration].copy()
            plot_df = self._reconstruct_mc_iteration_data(df_iter, filename, bit_count, metric_col)
            if plot_df is None:
                continue

            y = plot_df['y'].values
            x_labels = plot_df['label_index'] if 'label_index' in plot_df.columns else plot_df['label']
            
            # Calculate DNL and INL
            if len(y) > 1:
                if lsb is None:
                    lsb_ideal = (y[-1] - y[0]) / (len(y) - 1)
                else:
                    lsb_ideal = lsb
                if lsb_ideal == 0:
                    continue
                dnl = np.insert(np.diff(y) / lsb_ideal - 1, 0, 0)
                inl = np.cumsum(dnl)
                
                # Plot with transparency
                ax_dnl.plot(x_labels, dnl, marker='o', markersize=3, 
                           color=colors[idx], alpha=0.3, linewidth=0.8,
                           label=f"Iter {iteration}" if idx % (num_iterations // 5 + 1) == 0 else "")
                ax_inl.plot(x_labels, inl, marker='o', markersize=3,
                           color=colors[idx], alpha=0.3, linewidth=0.8,
                           label=f"Iter {iteration}" if idx % (num_iterations // 5 + 1) == 0 else "")
        
        # Formatting
        ax_dnl.set_ylabel("DNL (LSB)")
        if lsb is None:
            ax_dnl.set_title(f"DNL - All MC Realizations ({num_iterations} iterations, LSB=ideal)")
        else:
            if lsb is None:
                ax_dnl.set_title(f"DNL - All MC Realizations ({num_iterations} iterations, LSB=ideal)")
            else:
                ax_dnl.set_title(f"DNL - All MC Realizations ({num_iterations} iterations, LSB={lsb_ns} ns)")
        ax_dnl.grid(True, alpha=0.3)
        ax_dnl.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax_inl.set_ylabel("INL (LSB)")
        if lsb is None:
            ax_inl.set_title("INL - All MC Realizations (LSB=ideal)")
        else:
            if lsb is None:
                ax_inl.set_title("INL - All MC Realizations (LSB=ideal)")
            else:
                ax_inl.set_title(f"INL - All MC Realizations (LSB={lsb_ns} ns)")
        ax_inl.set_xlabel("Digital Code ($b_4b_3b_2b_1b_0$)")
        ax_inl.grid(True, alpha=0.3)
        ax_inl.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax_dnl.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='xx-small')
        plt.tight_layout()
        
        save_path = self._get_save_path(filename, "MC_Linearity", f"mc_all_iterations_lsb{lsb_ns}ns")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

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
        
        if has_bit_cols and not has_x_y_cols:
            plot_df = self._reconstruct_mc_code_sweep(df, filename, bit_count)
        else:
            plot_df = self._reconstruct_digital_data(df, filename, signal_name, bit_count)
        
        if plot_df is None: return

        y = plot_df['y'].values
        lsb_ideal = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
        dnl = np.insert(np.diff(y) / lsb_ideal - 1, 0, 0)
        inl = np.cumsum(dnl)

        # Create figure with improved styling
        fig, (ax1, ax2) = self._create_figure(nrows=2, ncols=1, figsize=(11, 9), sharex=True)
        
        # DNL plot
        ax1.plot(plot_df['label'], dnl, marker='o', color=self.colors['secondary'], 
                 linewidth=2.5, markersize=5, label='DNL')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self._format_plot_labels(ax1, ylabel='DNL (LSB)', 
                                title=f"{bit_count}-Bit Linearity Analysis: {signal_name}")
        self._apply_grid_styling(ax1)
        ax1.legend(fontsize=10, loc='upper right')
        
        # INL plot
        ax2.plot(plot_df['label'], inl, marker='o', color=self.colors['primary'], 
                 linewidth=2.5, markersize=5, label='INL')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        self._format_plot_labels(ax2, xlabel=f'Digital Code (LSB={lsb_ideal:.2e})', 
                                ylabel='INL (LSB)')
        self._apply_grid_styling(ax2)
        ax2.legend(fontsize=10, loc='upper right')
        
        # Rotate x-axis labels for readability
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self._get_save_path(filename, signal_name, "linearity")
        self._save_figure(fig, save_path)

    def plot_direct_csv_sweep(self, filename, y_col_idx=1, remove_code=True):
        """
        Plot delay/power vs code for direct CSV files (X, Y columns).
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
        codes_to_remove = original_length // 2 + 1
        
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
        ax1.plot(codes, y_to_plot, marker='o', color=self.colors['primary'], 
                 linewidth=2.5, markersize=4, label='Measured')
        
        self._format_plot_labels(ax1, 
                                xlabel='Digital Code',
                                ylabel=f"Relative {metric_info['name']} ({metric_info['prefix']}{metric_info['unit']})",
                                title=f"{path.stem.replace('_', ' ').title()} - Sweep"
        )
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels)
        ax1.legend(fontsize=10)
        self._apply_grid_styling(ax1)
        
        plt.tight_layout()
        save_path1 = self._get_save_path(filename, metric_info['name'], "sweep_decimal")
        self._save_figure(fig1, save_path1)
        
        # ===== PLOT 2: DNL and INL =====
        y = y_data.values
        lsb_ideal = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1e-9
        dnl = np.insert(np.diff(y) / lsb_ideal - 1, 0, 0)
        inl = np.cumsum(dnl)
        
        fig2, (ax2, ax3) = self._create_figure(nrows=2, ncols=1, figsize=(11, 9), sharex=True)
        
        # DNL plot
        ax2.plot(codes, dnl, marker='o', color=self.colors['secondary'], 
                linewidth=2.5, markersize=4, label='DNL')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self._format_plot_labels(ax2, 
                                ylabel='DNL (LSB)',
                                title=f"Differential Non-Linearity (LSB={lsb_ideal:.2e})"
        )
        ax2.legend(fontsize=10)
        self._apply_grid_styling(ax2)
        
        # INL plot
        ax3.plot(codes, inl, marker='o', color=self.colors['primary'], 
                linewidth=2.5, markersize=4, label='INL')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        self._format_plot_labels(ax3, 
                                xlabel='Digital Code',
                                ylabel='INL (LSB)',
                                title='Integral Non-Linearity'
        )
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(tick_labels)
        ax3.legend(fontsize=10)
        self._apply_grid_styling(ax3)
        
        plt.tight_layout()
        save_path2 = self._get_save_path(filename, metric_info['name'], "linearity_decimal")
        self._save_figure(fig2, save_path2)


    def plot_histogram(self, filenames):
        """
        Plots histograms for Monte Carlo data. 
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
            # You can replace 31 with a dynamic bit_count if available
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

        # --- Plotting Loop ---
        # If we have 2 metrics (DR/Res), we'll create two subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6 * len(metrics)))
        if len(metrics) == 1: axes = [axes]

        for ax, (label, data) in zip(axes, metrics.items()):
            ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            mean, std = data.mean(), data.std()
            
            ax.axvline(mean, color='red', linestyle='-', label=f'Mean: {mean:.3e}')
            ax.axvline(mean + std, color='orange', linestyle='--', label=f'Std: {std:.3e}')
            ax.axvline(mean - std, color='orange', linestyle='--')
            ax.axvspan(mean - std, mean + std, color='orange', alpha=0.1)
            
            ax.set_title(rf"Monte Carlo: {label}\n($\mu$={mean:.3e}, $\sigma$={std:.3e})")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self._get_save_path(filenames[0], "MC_Analysis", save_suffix)
        plt.savefig(save_path)
        plt.close()

    def plot_corner_temperature_sweep(self, filename):
        """Plots metric vs temperature for different PVT corners."""
        df, path = self.load_data(filename)
        if df is None: return None
        x_cols = [c for c in df.columns if c.endswith(' X')]
        plt.figure()
        corner_pattern = re.compile(r"top_(\w+)")
        for x_col in x_cols:
            y_col = x_col[:-2] + ' Y'
            if y_col in df.columns:
                match = corner_pattern.search(x_col)
                label = match.group(1).upper() if match else x_col
                plt.plot(df[x_col], df[y_col], marker='o', label=label)
        plt.xlabel(r"Temperature [$^{\circ}$C]"); plt.ylabel(path.stem.split('_')[0].capitalize())
        plt.title(f"Corner Sweep: {path.name}")
        plt.legend(title="Corners"); plt.grid(True); plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"corner_T")
        plt.savefig(save_path)

        plt.close()

    def plot_generic_sweep(self, filename):
        """Format: Single X and Y pair."""
        df, path = self.load_data(filename)
        x_label = self._get_axis_labels(filename) # <--- Dynamically gets 'Vdd', 'Cap', etc.

        if df is None: return None
        x_cols = [c for c in df.columns if c.endswith(' X')]
        y_cols = [c for c in df.columns if c.endswith(' Y')]
        if not x_cols or not y_cols: return None
        
        plt.figure()
        plt.plot(df[x_cols[0]], df[y_cols[0]], marker='o')
        plt.xlabel(x_label); plt.ylabel(y_cols[0].replace(' Y', ''))
        plt.title(f"Sweep: {path.stem}")
        plt.grid(True); plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"generic_sweep")
        plt.savefig(save_path)
    
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
            fig, axes = plt.subplots(num_signals, 1, sharex=True, figsize=(12, 3 * num_signals))
            if num_signals == 1: axes = [axes]
        else:
            plt.figure()
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
                label = label_match.group(1) if label_match else base_name

                ax.plot(x_vals[mask], y_vals[mask], label=label)
            
            ax.set_ylabel("Value")
            ax.set_title(f"Signal: {base_name}")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            if t_range:
                ax.set_xlim(t_range)

        plt.xlabel("Time (s)")
        plt.tight_layout()

        save_path = self._get_save_path(filename, base_name ,"trans")
        plt.savefig(save_path)
        
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
        corner_colors_std = {'SS': 'red', 'TT': 'blue', 'FF': 'green', 'SF': 'orange', 'FS': 'purple'}
        temp_styles_std = {80: '-', 0: '--', -55: ':'} 

        # --- Mode B: VDD is True (Color = Corner_Temp, Style = VDD) ---
        # Define your specific group colors here
        pvt_group_colors = {
            'SS_80': 'darkred', 'SS_0': 'red', 'SS_-55': 'salmon',
            'TT_80': 'darkblue', 'TT_0': 'blue', 'TT_-55': 'skyblue',
            'FF_80': 'darkgreen', 'FF_0': 'green', 'FF_-55': 'lime'
        }
        vdd_styles = ['-', '--', ':', '-.']

        def plot_metric(y_col, y_label, scale, unit, suffix):
            unique_corners = sorted(df['base_corner'].unique())
            
            if subfigure:
                fig, axes = plt.subplots(len(unique_corners), 1, figsize=(14, 5 * len(unique_corners)), sharex=True)
                if len(unique_corners) == 1: axes = [axes]
            else:
                plt.figure(figsize=(14, 8))
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
                    
                    lbl = f"{c_full}, {temp}C"
                    if VDD: lbl += f", {v_val}V"
                    lbl += f" | DR: {dr:.2f}{unit}"

                    # --- STYLING LOGIC ---
                    if VDD:
                        # Construct key like 'SS_80'
                        color_key = f"{base_corner}_{temp}"
                        line_color = pvt_group_colors.get(color_key, 'black')
                        line_style = v_style_map[v_val]
                    else:
                        line_color = corner_colors_std.get(base_corner, 'black')
                        line_style = temp_styles_std.get(temp, '-')

                    ax.plot(x_labels, y_vals, label=lbl, color=line_color, 
                            linestyle=line_style, marker='o', markersize=3, alpha=0.8)

                ax.set_title(f"Corner: {base_corner}" if subfigure else f"{y_label} vs Code")
                ax.set_ylabel(f"{y_label} ({unit})")
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='xx-small')

            plt.xlabel("Digital Code ($b_4b_3b_2b_1b_0$)")
            plt.xticks(rotation=90, fontsize=8)
            plt.tight_layout()
            
            final_suffix = f"{suffix}_subplots" if subfigure else suffix
            save_path = self._get_save_path(filename, y_label.replace(' ', '_'), final_suffix)
            plt.savefig(save_path); plt.close()

        plot_metric('delay', 'Delay', 1e9, 'ns', 'pvt_delay')
        plot_metric('P_avg', 'Average Power', 1e6, 'uW', 'pvt_power')

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
        corner_colors = {'SS': 'red', 'TT': 'blue', 'FF': 'green', 'SF': 'orange', 'FS': 'purple'}
        temp_styles = {80: '-', 0: '--', -55: ':'} 

        # 3. Dynamic Figure Setup
        if subplots:
            num_rows = len(unique_corners)
            fig, axes = plt.subplots(num_rows, 2, figsize=(16, 4 * num_rows), sharex=True)
            # Ensure axes is 2D even if only one corner exists
            if num_rows == 1: axes = np.expand_dims(axes, axis=0)
        else:
            fig, (ax_dnl, ax_inl) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
            # Duplicate the axes references so the loop logic remains the same
            axes = [[ax_dnl, ax_inl]] * len(unique_corners)

        # 4. Process Each Corner
        for i, base_corner in enumerate(unique_corners):
            # If subplots=False, every iteration uses the same ax_dnl/ax_inl
            # If subplots=True, each iteration uses a unique row
            cur_ax_dnl = axes[i][0]
            cur_ax_inl = axes[i][1]
            
            corner_group = df[df['base_corner'] == base_corner]
            
            for (corner_full, temp), group in corner_group.groupby(['Corner', 'temperature']):
                group = group.sort_values('code')
                y = group['delay'].values
                x_labels = group['code'].apply(lambda c: bin(c)[2:].zfill(5))
                
                lsb = (y.max() - y.min()) / (len(y) - 1)
                dnl = np.insert(np.diff(y) / lsb - 1, 0, 0)
                inl = np.cumsum(dnl)

                color = corner_colors.get(base_corner, 'black')
                style = temp_styles.get(temp, '-')
                lbl = f"{corner_full}, {temp}C"

                cur_ax_dnl.plot(x_labels, dnl, label=lbl, color=color, linestyle=style, marker='o', markersize=2, alpha=0.6)
                cur_ax_inl.plot(x_labels, inl, label=lbl, color=color, linestyle=style, marker='o', markersize=2, alpha=0.6)

            # Labels for Subplot Mode
            if subplots:
                cur_ax_dnl.set_title(f"DNL - {base_corner}")
                cur_ax_inl.set_title(f"INL - {base_corner}")
                cur_ax_inl.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='xx-small')

        # 5. Global Formatting
        if not subplots:
            axes[0][0].set_title("DNL (Differential Non-Linearity)")
            axes[0][1].set_title("INL (Integral Non-Linearity)")
            axes[0][0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='xx-small')

        for ax in fig.axes:
            ax.grid(True, alpha=0.3)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Digital Code ($b_4b_3b_2b_1b_0$)")
                ax.tick_params(axis='x', rotation=90, labelsize=8)

        plt.tight_layout()
        suffix = "pvt_linearity_split" if subplots else "pvt_linearity_combined"
        save_path = self._get_save_path(filename, "Linearity", suffix)
        plt.savefig(save_path); plt.close()

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
                "cs_delay_code_4bit.csv", "cs_delay_code_5bit.csv", 
                "cs_delay_code_5bit_nonidealcurs.csv", "cs_delay_code_5bit_coarse.csv",
                "cs_power_code_4bit.csv", "cs_power_code_5bit.csv", 
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
                "cs_delay_code_4bit.csv", "cs_delay_code_5bit.csv",
                "cs_delay_code_5bit_coarse.csv", "cs_delay_code_5bit_nonidealcurs.csv",
                "cs_delay_code_5bit_counter.csv"
            ],
            "action": lambda plotter, f: plotter.plot_linearity(f)
        },
        "constant_slope_mc_linearity": {
            "description": "Constant slope MC linearity per iteration",
            "files": ["cs_delay_code_5bit_mc.csv"],
            "action": lambda plotter, f: plotter.plot_mc_linearity_per_iteration(f, max_iterations=20)
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
        "constant_slope_digital_sweep_mc": {
            "description": "Constant slope digital sweep MC all iterations",
            "files": ["cs_delay_code_5bit_mc.csv"],
            "action": lambda plotter, f: plotter.plot_digital_sweep(f, max_iterations=20)
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


# --- Usage ---
if __name__ == "__main__":
    plotter = CadencePlotter(base_dir="results_cadence")
    
    # Option 1: Plot everything
    run_plots(plotter, plot_all=True)
    
    # Option 2: Plot only specific tasks (uncomment to use)
    # selected = [
    #     "constant_slope_sweeps",
    #     "constant_slope_linearity",
    #     "constant_slope_direct_csv"
    # ]
    # run_plots(plotter, plot_all=False, selected_tasks=selected)