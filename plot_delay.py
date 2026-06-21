from os import remove

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import numpy as np
import argparse
from pathlib import Path

from plot_style import apply_science_style, _multi_panel_figsize, maybe_title, maybe_suptitle

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
        

    # ========================================================================
    # HELPER METHODS - Reduce Redundancy
    # ========================================================================
    
    def _save_figure(self, fig, save_path, dpi=300, bbox_inches='tight'):
        """Unified figure saving with consistent parameters."""
        fig.canvas.draw()  # Ensure figure is fully rendered
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=0.05)
        print(f"  Saved: {save_path.name}")
        plt.close(fig)  # Close specific figure
    
    def _create_figure(self, nrows=1, ncols=1, sharex=False, sharey=False):
        """Unified figure creation with consistent parameters."""
        if nrows == 1 and ncols == 1:
            fig, ax = plt.subplots()
            return fig, ax
        return plt.subplots(
            nrows,
            ncols,
            figsize=_multi_panel_figsize(nrows, ncols),
            sharex=sharex,
            sharey=sharey,
        )
    
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
            maybe_title(ax, title)
    
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

    def _extract_phase_noise_sweeps(self, df):
        """Extract phase noise sweeps from X/Y columns and label by bit code."""
        x_cols = [c for c in df.columns if str(c).endswith(' X') or str(c).endswith('X')]
        sweeps = []

        for x_col in x_cols:
            x_col_str = str(x_col)
            y_col = x_col_str[:-1] + 'Y' if x_col_str.endswith('X') else x_col_str[:-2] + ' Y'
            if y_col not in df.columns:
                continue

            header_bits = re.findall(r"b(\d+)=(\d+)", x_col_str)
            if header_bits:
                bit_dict = {int(idx): int(val) for idx, val in header_bits}
                max_bit = max(bit_dict.keys())
                bits = [bit_dict.get(i, 0) for i in range(max_bit, -1, -1)]
                label = "".join(str(b) for b in bits)
                if all(b == 0 for b in bits):
                    color_key = "all0"
                elif all(b == 1 for b in bits):
                    color_key = "all1"
                else:
                    color_key = label
            else:
                label = "code"
                color_key = label

            x_vals = pd.to_numeric(df[x_col_str], errors='coerce')
            y_vals = pd.to_numeric(df[y_col], errors='coerce')
            mask = x_vals.notna() & y_vals.notna()
            if not mask.any():
                continue

            sweeps.append({
                "label": label,
                "color_key": color_key,
                "x": x_vals[mask].values,
                "y": y_vals[mask].values,
            })

        return sweeps

    def _compute_jitter_rms(self, freq_hz, phase_noise_dbc_hz, carrier_hz):
        """Compute RMS jitter from phase noise over the full frequency span."""
        if carrier_hz is None or carrier_hz <= 0:
            return None

        freq_arr = np.asarray(freq_hz, dtype=float)
        pn_arr = np.asarray(phase_noise_dbc_hz, dtype=float)
        mask = np.isfinite(freq_arr) & np.isfinite(pn_arr) & (freq_arr > 0)
        if not mask.any():
            return None

        f = freq_arr[mask]
        pn = pn_arr[mask]
        order = np.argsort(f)
        f = f[order]
        pn = pn[order]

        # Convert L(f) [dBc/Hz] to phase noise PSD Sphi [rad^2/Hz]
        s_phi = 2.0 * np.power(10.0, pn / 10.0)
        integral = np.trapz(s_phi, f)
        jitter_rms = np.sqrt(integral) / (2.0 * np.pi * carrier_hz)
        return float(jitter_rms)

    def plot_phase_noise_comparison(self, cs_filename, vs_filename=None, title=None, log_x=True, carrier_hz=None, label_a=None, label_b=None):
        """
        Plots phase noise sweeps overlaid with code-based colors and standardized line styles.
        File A (Pre) uses a dashed line, File B (Post) uses a solid line.
        Applies proper LaTeX typesetting to underscores and exports directly to a high-fidelity PDF.
        """
        if not cs_filename:
            print("Warning: Provide at least one phase noise file.")
            return

        cs_df, _ = self.load_data(cs_filename)
        vs_df = None
        if vs_filename:
            vs_df, _ = self.load_data(vs_filename)
        if cs_df is None:
            return

        cs_sweeps = self._extract_phase_noise_sweeps(cs_df)
        vs_sweeps = self._extract_phase_noise_sweeps(vs_df) if vs_df is not None else []
        if not cs_sweeps and not vs_sweeps:
            print("Warning: No phase noise sweeps found.")
            return

        # Ensure uniform Pre/Post label strings
        label_a = "Pre" if not label_a else label_a.strip()
        label_b = "Post" if not label_b else label_b.strip()
        
        # Sort keys to pair identical sweep conditions to the exact same color mapping index
        all_keys = sorted({s.get("color_key", s["label"]) for s in cs_sweeps + vs_sweeps})
        cmap = plt.get_cmap("tab10")
        label_colors = {
            key: cmap(idx % cmap.N)
            for idx, key in enumerate(all_keys)
        }
        
        fig, ax = self._create_figure() if hasattr(self, '_create_figure') else plt.subplots(figsize=(14, 5))

        # Internal helper to handle conversion of underscore-separated labels to proper LaTeX format
        def latex_format_string(text_str):
            if "_" in text_str:
                words = text_str.split()
                for w_idx, word in enumerate(words):
                    if "_" in word:
                        punctuation = ""
                        if word[-1] in [".", ",", ":", ";", ")", " "]:
                            punctuation = word[-1]
                            word = word[:-1]
                        
                        if "_" in word:
                            base, subscript = word.split("_", 1)
                            words[w_idx] = f"${base}_{{{{\\text{{{subscript}}}}}}}${punctuation}"
                return " ".join(words)
            return text_str

        # 1. Plot File A: Pre-layout Curves (Dashed Line)
        for sweep in cs_sweeps:
            jitter_rms = self._compute_jitter_rms(sweep["x"], sweep["y"], carrier_hz)
            jitter_label = f" | RMS={jitter_rms * 1e12:.2f} ps" if jitter_rms is not None else ""
            color_key = sweep.get("color_key", sweep["label"])
            curve_color = label_colors.get(color_key, cmap(0))
            
            clean_sweep_label = latex_format_string(sweep['label'])
            ax.plot(
                sweep["x"],
                sweep["y"],
                color=curve_color,
                linestyle='--',
                linewidth=2.2,
                label=f"{label_a} {clean_sweep_label}{jitter_label}",
            )

        # 2. Plot File B: Post-layout Curves (Solid Line)
        for sweep in vs_sweeps:
            jitter_rms = self._compute_jitter_rms(sweep["x"], sweep["y"], carrier_hz)
            jitter_label = f" | RMS={jitter_rms * 1e12:.2f} ps" if jitter_rms is not None else ""
            color_key = sweep.get("color_key", sweep["label"])
            curve_color = label_colors.get(color_key, cmap(1))
            
            clean_sweep_label = latex_format_string(sweep['label'])
            ax.plot(
                sweep["x"],
                sweep["y"],
                color=curve_color,
                linestyle='-',
                linewidth=2.2,
                label=f"{label_b} {clean_sweep_label}{jitter_label}",
            )

        if log_x:
            ax.set_xscale('log')

        plot_title = title or "Phase Noise Comparison"
        plot_title = latex_format_string(plot_title)
        
        if hasattr(self, '_format_plot_labels'):
            self._format_plot_labels(
                ax,
                xlabel="Offset Frequency (Hz)",
                ylabel="Phase Noise (dBc/Hz)",
                title=plot_title,
            )
        else:
            ax.set_xlabel("Offset Frequency (Hz)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Phase Noise (dBc/Hz)", fontsize=11, fontweight='bold')
            ax.set_title(plot_title, fontsize=12, fontweight='bold')

        # 3. Corrected crisp black-bordered font-15 legend configuration
        ax.legend(
            loc='best', 
            fontsize=5, 
            framealpha=1.0, 
            edgecolor='black', 
            facecolor='white',
            labelcolor='black'
        )
        
        if hasattr(self, '_apply_grid_styling'):
            self._apply_grid_styling(ax, alpha=0.35)
        else:
            ax.grid(True, which="both", linestyle='--', alpha=0.35)

        plt.tight_layout()
        target_dir = self.plot_dir / "phase_noise"
        target_dir.mkdir(exist_ok=True)
        
        # 4. Save path conversion explicitly formatted to output as a vector .pdf
        if vs_filename:
            save_path = target_dir / (
                f"phase_noise_{self._sanitize(Path(cs_filename).stem)}"
                f"_vs_{self._sanitize(Path(vs_filename).stem)}.pdf"
            )
        else:
            save_path = target_dir / f"phase_noise_{self._sanitize(Path(cs_filename).stem)}.pdf"
            
        if hasattr(self, '_save_figure'):
            self._save_figure(fig, save_path)
        else:
            plt.savefig(save_path, dpi=300)
            print(f"Saved compared phase noise diagram to: {save_path}")
            plt.show()

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

        fig, ax = self._create_figure()

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

        fig, (ax_dnl, ax_inl) = self._create_figure(nrows=2, ncols=1, sharex=True)

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
        maybe_suptitle(ax_dnl,clean_title, fontsize=14, fontweight='bold', pad=15)
        ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_dnl.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='±0.5 LSB')
        ax_dnl.axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax_dnl.set_ylim(-0.7, 0.7)
        ax_dnl.legend(fontsize=10, loc='upper right', framealpha=0.95)
        self._apply_grid_styling(ax_dnl, alpha=0.35)

        ax_inl.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
        maybe_suptitle(ax_inl, "Integral Non-Linearity (INL)", fontsize=14, fontweight='bold', pad=15)
        ax_inl.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
        ax_inl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_inl.legend(fontsize=10, loc='upper right', framealpha=0.95)
        self._apply_grid_styling(ax_inl, alpha=0.35)

        plt.tight_layout()
        save_path = self._get_save_path(filename, "MC_Linearity", f"mcparamset_linearity_{len(sweeps)}")
        self._save_figure(fig, save_path)

    def plot_average_power_by_code(self, filename=None, file_a=None, file_b=None, 
                                   start_time=2e-8, window_size=5e-8, num_codes=256, 
                                   remove_code=True, P_static=0, **kwargs):
        """
        Plots and overlays average power consumption by digital code for both Pre-layout (--file_a)
        and Post-layout (--file_b) datasets. 
        File A (Pre) uses a dashed line, File B (Post) uses a solid line.
        """
        df_a = None
        df_b = None
        
        # 1. Load File A and File B dataframes explicitly
        if file_a:
            df_a, _ = self.load_data(file_a)
        if file_b:
            df_b, _ = self.load_data(file_b)
            
        # Fallback to single filename argument if separate files are not provided
        if df_a is None and df_b is None and filename:
            if isinstance(filename, (str, Path)):
                df_b, _ = self.load_data(filename)
            else:
                df_b = filename

        # Helper internal function to window-integrate power data over sequential codes
        def compute_windowed_power(df, add):
            if df is None:
                return None
            
            # Align time and power columns safely
            time_col = df.columns[0]
            power_col = df.columns[1]
            local_df = df.rename(columns={time_col: 'Time', power_col: 'Power'})
            
            all_averages = []
            for i in range(num_codes):
                t_start = start_time + add + i * window_size
                t_end = start_time + add + (i + 1) * window_size
                
                mask = (local_df['Time'] >= t_start) & (local_df['Time'] < t_end)
                window_data = local_df[mask]
                
                if not window_data.empty:
                    times = window_data['Time'].values
                    powers = np.abs(window_data['Power'].values)
                    # Trapezoidal integration
                    energy = np.sum((times[1:] - times[:-1]) * (powers[1:] + powers[:-1]) / 2.0)
                    actual_duration = times[-1] - times[0]
                    avg_p = energy / actual_duration if actual_duration > 0 else window_data['Power'].mean()
                else:
                    avg_p = 0.0
                all_averages.append(avg_p)
            return all_averages

        averages_a = compute_windowed_power(df_a, add = -0.001e-9)
        averages_b = compute_windowed_power(df_b, add = 0.00843e-9)

        # Identify middle index constraints for compression logic
        remove_idx = num_codes // 2 if num_codes > 0 else None

        def process_plotting_arrays(raw_averages):
            if raw_averages is None:
                return None
            
            plot_averages = []
            if remove_code and remove_idx is not None:
                for i in range(num_codes):
                    if i == remove_idx:
                        continue
                    plot_averages.append(raw_averages[i])
            else:
                plot_averages = list(raw_averages)
                
            # Convert to micro-watts (uW) and apply static power correction offset
            return (np.array(plot_averages) - P_static) * 1e6

        y_plot_a = process_plotting_arrays(averages_a)
        y_plot_b = process_plotting_arrays(averages_b)

        # Build clean index codes for horizontal representation
        sample_y = y_plot_b if y_plot_b is not None else y_plot_a
        if sample_y is None:
            print("No valid power data found to process.")
            return
        plot_codes = list(range(len(sample_y)))

        # Create figure using your standard workspace preferences
        fig, ax = self._create_figure() if hasattr(self, '_create_figure') else plt.subplots(figsize=(14, 5))
        
        # Use a high-contrast standard color cycle trace index
        curve_color = plt.cm.tab10(0)
        curve_color_b = plt.cm.tab10(1)

        # 2. Plot Pre-layout array profile (Dashed Line)
        if y_plot_a is not None:
            ax.plot(plot_codes, y_plot_a, color=curve_color, linestyle='--', linewidth=2, label='Pre')

        # 3. Plot Post-layout array profile (Solid Line)
        if y_plot_b is not None:
            ax.plot(plot_codes, y_plot_b, color=curve_color_b, linestyle='-', linewidth=2, label='Post')

        # Formatting titles safely using internal format managers
        target_name = file_b if file_b else (file_a if file_a else filename)
        thesis_title = self._get_thesis_title(target_name, "sweep") if hasattr(self, '_get_thesis_title') else "Power Consumption Sweep"
        clean_title = self._format_title(thesis_title) if hasattr(self, '_format_title') else thesis_title
        
        if hasattr(self, '_format_plot_labels'):
            self._format_plot_labels(
                ax,
                xlabel="Digital Code",
                ylabel="$P_{\\text{tot}}$ ($\\mu W$)",
                title=clean_title
            )
        else:
            ax.set_xlabel("Digital Code")
            ax.set_ylabel("$P_{\\text{tot}}$ ($\\mu W$)")
            ax.set_title(clean_title, fontweight='bold')

        # 4. Apply clean black-bordered crisp legend configuration
        ax.legend(
            loc='best', 
            fontsize=12, 
            framealpha=1.0, 
            edgecolor='black', 
            facecolor='white',
            labelcolor='black'
        )

        if hasattr(self, '_apply_grid_styling'):
            self._apply_grid_styling(ax, alpha=0.35)
        else:
            ax.grid(True, linestyle='--', alpha=0.35)
            
        plt.tight_layout()

        # Save directly to the publication-ready vector PDF profile extension format
        if hasattr(self, '_get_save_path') and hasattr(self, '_save_figure'):
            save_path = self._get_save_path(target_name, "Avg_Power", "avg_power_by_code")
            # Force conversion of file paths to output as a vector PDF
            save_path = Path(save_path).with_suffix('.pdf')
            self._save_figure(fig, save_path)
        else:
            out_path = self.plot_dir / "avg_power_by_code.pdf"
            plt.savefig(out_path, dpi=300)
            print(f"Saved compared power sweep plots to: {out_path}")
            plt.show()

    def plot_digital_sweep(self, filename=None, file_a=None, file_b=None, 
                           y_col_idx=1, remove_code=False, num_codes=256, **kwargs):
        """
        Plots and overlays sweeping metrics across digital codes for Pre-layout (--file_a)
        and Post-layout (--file_b) files.
        File A (Pre) uses a dashed line, File B (Post) uses a solid line.
        Exports directly as a vector PDF with LaTeX formatting on titles and labels.
        """
        df_a = None
        df_b = None
        
        # 1. Load File A and File B dataframes explicitly
        if file_a:
            df_a, _ = self.load_data(file_a)
        if file_b:
            df_b, _ = self.load_data(file_b)
            
        # Fallback to single filename argument if separate files are not provided
        if df_a is None and df_b is None and filename:
            if isinstance(filename, (str, Path)):
                df_b, _ = self.load_data(filename)
            else:
                df_b = filename

        # Helper internal function to clean and compress individual sweeps
        def process_sweep_dataframe(df):
            if df is None:
                return None, None
            
            # Detect target sweeping value column cleanly
            y_col = df.columns[y_col_idx] if len(df.columns) > y_col_idx else df.columns[-1]
            
            # Read unique rows to map digital code tracking values
            y_vals = pd.to_numeric(df[y_col], errors='coerce').dropna().values
            raw_codes = list(range(len(y_vals)))
            
            remove_idx = num_codes // 2 if num_codes > 0 else None
            
            plot_y = []
            if remove_code and remove_idx is not None and len(y_vals) >= num_codes:
                for i in range(min(num_codes, len(y_vals))):
                    if i == remove_idx:
                        continue
                    plot_y.append(y_vals[i])
                plot_codes = list(range(len(plot_y)))
            else:
                plot_y = list(y_vals[:num_codes])
                plot_codes = list(raw_codes[:num_codes])
                
            return plot_codes, np.array(plot_y)

        codes_a, y_plot_a = process_sweep_dataframe(df_a)
        codes_b, y_plot_b = process_sweep_dataframe(df_b)

        # Create figure using your standard workspace preferences
        fig, ax = self._create_figure() if hasattr(self, '_create_figure') else plt.subplots(figsize=(14, 5))
        
        # Use two different distinct colors from the high-contrast palette
        curve_color = plt.cm.tab10(0)   # Blue for Pre
        curve_color_b = plt.cm.tab10(1) # Orange for Post

        # 2. Plot Pre-layout Sweep Profile (Dashed Line)
        if y_plot_a is not None:
            ax.plot(codes_a, y_plot_a, color=curve_color, linestyle='--', linewidth=2.6, label='Pre')

        # 3. Plot Post-layout Sweep Profile (Solid Line)
        if y_plot_b is not None:
            ax.plot(codes_b, y_plot_b, color=curve_color_b, linestyle='-', linewidth=2.6, label='Post')

        # Formatting titles safely using internal format managers
        target_name = file_b if file_b else (file_a if file_a else filename)
        thesis_title = self._get_thesis_title(target_name, "sweep") if hasattr(self, '_get_thesis_title') else "Digital Code Sweep Analysis"
        clean_title = self._format_title(thesis_title) if hasattr(self, '_format_title') else thesis_title
        
        # Parse and format variable tokens with underscores inside the text title using LaTeX
        if "_" in clean_title:
            words = clean_title.split()
            for w_idx, word in enumerate(words):
                if "_" in word:
                    # Strip trailing punctuation marks if present
                    punctuation = ""
                    if word[-1] in [".", ",", ":", ";", ")", " "]:
                        punctuation = word[-1]
                        word = word[:-1]
                    
                    if "_" in word:
                        base, subscript = word.split("_", 1)
                        words[w_idx] = f"${base}_{{{{\\text{{{subscript}}}}}}}${punctuation}"
            clean_title = " ".join(words)

        # Pull Y-axis label context dynamically from dataframe column headers
        sample_df = df_b if df_b is not None else df_a
        y_label_text = sample_df.columns[y_col_idx] if sample_df is not None else "Value"
        y_label_text = y_label_text.replace('/', '').replace(' Y', '').strip()

        # Parse and format variable tokens with underscores inside the Y-label using LaTeX
        if "_" in y_label_text:
            words = y_label_text.split()
            for w_idx, word in enumerate(words):
                if "_" in word:
                    punctuation = ""
                    if word[-1] in [".", ",", ":", ";", ")", " "]:
                        punctuation = word[-1]
                        word = word[:-1]
                    
                    if "_" in word:
                        base, subscript = word.split("_", 1)
                        words[w_idx] = f"${base}_{{{{\\text{{{subscript}}}}}}}${punctuation}"
            y_label_text = " ".join(words)

        if hasattr(self, '_format_plot_labels'):
            self._format_plot_labels(
                ax,
                xlabel="Digital Code",
                ylabel=y_label_text,
                title=clean_title
            )
        else:
            ax.set_xlabel("Digital Code", fontsize=11, fontweight='bold')
            ax.set_ylabel(y_label_text, fontsize=11, fontweight='bold')
            ax.set_title(clean_title, fontsize=12, fontweight='bold')

        # 4. Corrected crisp black-bordered font-15 legend configuration
        ax.legend(
            loc='best', 
            fontsize=15, 
            framealpha=1.0, 
            edgecolor='black', 
            facecolor='white',
            labelcolor='black'
        )

        if hasattr(self, '_apply_grid_styling'):
            self._apply_grid_styling(ax, alpha=0.35)
        else:
            ax.grid(True, linestyle='--', alpha=0.35)
            
        plt.tight_layout()

        # Save directly to the publication-ready vector PDF profile extension format
        if hasattr(self, '_get_save_path') and hasattr(self, '_save_figure'):
            save_path = self._get_save_path(target_name, "Digital_Sweep", "digital_sweep_by_code")
            save_path = Path(save_path).with_suffix('.pdf')
            self._save_figure(fig, save_path)
        else:
            out_path = self.plot_dir / "digital_sweep_by_code.pdf"
            plt.savefig(out_path, dpi=300)
            print(f"Saved compared digital sweep to: {out_path}")
            
    def plot_mc_sweep_all_realizations(self, filename, max_realizations=200,remove_code=False):
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
        
        data_clean = data.iloc[:, :num_realizations]
        if remove_code:
            data_clean, removed_idx = self._remove_middle_code_rows(data_clean)
        else:
            removed_idx = None
        codes = np.arange(data_clean.shape[0])
        
        print(f"Processing {num_realizations} sweep curves with {data_clean.shape[0]} codes per curve")
        if removed_idx is not None:
            print(f"Removed redundant middle code at index {removed_idx}")
        
        # Create figure
        fig, ax = self._create_figure()
        
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
        maybe_title(ax, clean_title, fontsize=14, fontweight='bold', pad=15)
        self._apply_grid_styling(ax)
        
        plt.tight_layout()
        
        save_path = self._get_save_path(filename, "MC_Sweep", f"mc_all_{num_realizations}_sweeps")
        self._save_figure(fig, save_path)

    def plot_mc_linearity_all_realizations_xy(self, filename, max_realizations=200, remove_code=False):
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
        if remove_code:
            data_clean, removed_idx = self._remove_middle_code_rows(data_subset)
        else:
            data_clean = data_subset
            removed_idx = None

        print(f"Processing {num_realizations} iterations with {data_clean.shape[0]} codes per iteration")
        if removed_idx is not None:
            print(f"Removed redundant middle code at index {removed_idx}")
        
        # Create figure with 2 subplots (DNL and INL)
        fig, (ax_dnl, ax_inl) = self._create_figure(nrows=2, ncols=1, sharex=True)
        
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
                        alpha=0.2, linewidth=1.2)
            ax_inl.plot(codes, inl,
                        alpha=0.2, linewidth=1.2)
        
        # Formatting
        thesis_title_dnl = self._get_thesis_title(filename, "mc_linearity_all")
        clean_title = self._format_title(thesis_title_dnl)
        
        ax_dnl.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
        maybe_suptitle(ax_dnl, clean_title, fontsize=14, fontweight='bold', pad=15)
        self._apply_grid_styling(ax_dnl, alpha=0.35)
        ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_dnl.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.6, label='±0.5 LSB')
        ax_dnl.axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax_dnl.set_ylim(-0.7, 0.7)
        ax_dnl.legend(fontsize=10, loc='upper right', framealpha=0.95)
        
        ax_inl.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
        maybe_suptitle(ax_inl, "Integral Non-Linearity (INL)", fontsize=14, fontweight='bold', pad=15)
        ax_inl.set_xlabel("Digital Code", fontsize=12, fontweight='bold')
        self._apply_grid_styling(ax_inl, alpha=0.35)
        ax_inl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_inl.legend(fontsize=10, loc='upper right', framealpha=0.95)

        plt.tight_layout()

        save_path = self._get_save_path(filename, "MC_Linearity", f"mc_all_{num_realizations}_dnl_inl")
        save_path = Path(save_path).with_suffix('.pdf')
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
        fig, (ax_dnl, ax_inl) = self._create_figure(nrows=2, ncols=1, sharex=True)
        
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
        maybe_suptitle(ax_dnl, f"DNL - All {num_iterations} MC Realizations (LSB=ideal)", fontsize=14, fontweight='bold')
        self._apply_grid_styling(ax_dnl)
        ax_dnl.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        ax_dnl.axhline(y=1, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='±1 LSB')
        ax_dnl.axhline(y=-1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_dnl.legend(fontsize=10, loc='upper right')
        
        ax_inl.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
        maybe_suptitle(ax_inl, f"INL - All {num_iterations} MC Realizations (LSB=ideal)", fontsize=14, fontweight='bold')
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
        fig, (ax1, ax2) = self._create_figure(nrows=2, ncols=1, sharex=True)
        
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
        fig1, ax1 = self._create_figure()
        ax1.plot(codes, y_to_plot, marker=None, 
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
        
        fig2, (ax2, ax3) = self._create_figure(nrows=2, ncols=1, sharex=True)
        
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


    def plot_histogram(self, filenames, remove_code=True, max_realizations=200):
        """
        Plots histograms for Monte Carlo data with improved styling.
        If a single MC X/Y dataset is provided, it calculates dynamic range and
        resolution per realization.
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

        # --- Case A: Single MC X/Y file (Calculate DR and Resolution) ---
        if len(datasets) == 1:
            df = datasets[0]
            x_cols = [c for c in df.columns if c.endswith(' X')]
            if x_cols:
                metric_info = self._get_metric_info(filenames[0])
                dr_values = []
                res_values = []

                for x_col in x_cols:
                    y_col = x_col[:-2] + ' Y'
                    if y_col not in df.columns:
                        continue

                    y_vals = pd.to_numeric(df[y_col], errors='coerce').dropna().values
                    if len(y_vals) < 2:
                        continue

                    if remove_code:
                        y_df = pd.DataFrame({'y': y_vals})
                        y_df, _ = self._remove_middle_code_rows(y_df)
                        if y_df is not None:
                            y_vals = y_df['y'].values

                    if len(y_vals) < 2:
                        continue

                    dr = float(np.max(y_vals) - np.min(y_vals))
                    res = dr / (len(y_vals) - 1)
                    dr_values.append(dr)
                    res_values.append(res)

                    if len(dr_values) >= max_realizations:
                        break

                if not dr_values or not res_values:
                    return

                metrics = {
                    f"Dynamic Range (n{metric_info['unit']})": np.array(dr_values) * 1e9,
                    f"Resolution (p{metric_info['unit']})": np.array(res_values) * 1e12,
                }
                save_suffix = "mc_derived_metrics"
            else:
                data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
                metrics = {df.columns[0]: data}
                save_suffix = "mc_standard"

        # --- Case B: Two Files (Calculate DR and Resolution) ---
        elif len(datasets) == 2:
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
        
        # --- Case C: Single File (Standard Histogram) ---
        else:
            data = pd.to_numeric(datasets[0].iloc[:, 0], errors='coerce').dropna()
            metrics = {datasets[0].columns[0]: data}
            save_suffix = "mc_standard"

        # --- Plotting Loop with Enhanced Styling ---
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=_multi_panel_figsize(num_metrics, 1))
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
            
            clean_label = self._format_title(label)
            if clean_label.lower() == "resolution (ps)":
                scale_factor = 1e12
                unit = "ps"
            else:
                scale_factor = 1e9
                unit = "ns"

            # Calculate statistics
            mean, std = data.mean(), data.std()
            min_val, max_val = data.min(), data.max()
            
            # Plot vertical lines for statistics
            ax.axvline(mean, color='#d62728', linestyle='-', linewidth=2.5,
                      label=f'Mean: {mean:.2f} {unit}', alpha=0.9)
            # ax.axvline(median, color='#2ca02c', linestyle='--', linewidth=2,
            #           label=f'Median: {median:.3e}', alpha=0.8)

            # Highlight +/- 3 sigma region
            three_sigma = 3 * std
            ax.axvline(mean + three_sigma, color='#ff7f0e', linestyle=':', linewidth=2,
                      label=rf'$\pm 3 \sigma$: {three_sigma:.2f} {unit}', alpha=0.85)
            ax.axvline(mean - three_sigma, color='#ff7f0e', linestyle=':', linewidth=2, alpha=0.85)
            ax.axvspan(mean - three_sigma, mean + three_sigma, color='#ff7f0e', alpha=0.08)
            
            # Formatting
            thesis_title = self._get_thesis_title(filenames[0], "mc_distribution")
            clean_label = self._format_title(label)
            
            maybe_title(ax, thesis_title, fontsize=14, fontweight='bold', pad=15)
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
        
        fig, ax = plt.subplots()
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
        maybe_title(ax, thesis_title, fontsize=14, fontweight='bold', pad=15)
        
        ax.legend(title="Corners", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"corner_T")
        self._save_figure(fig, save_path)

    def plot_corner_linearity_xy(self, filename, remove_code=True):
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

        corners = ["SS", "TT", "FF", "FS", "SF"]
        fig = {}
        ax_dnl = {}
        ax_inl = {}
        fig_sweep = {}
        ax_sweep = {}

        for corner in corners:
            fig[corner], (ax_dnl[corner], ax_inl[corner]) = plt.subplots(
                2,
                1,
                figsize=_multi_panel_figsize(2, 1),
            )
            fig[corner].patch.set_facecolor('white')
            fig[corner].patch.set_alpha(1.0)
            
            fig_sweep[corner], ax_sweep[corner] = self._create_figure()

        # Use a high-contrast standard color cycle trace index
        curve_color = plt.cm.tab10(0)
        curve_color_b = plt.cm.tab10(1)
        color_cycle = [
            plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2), plt.cm.tab10(3), plt.cm.tab10(4), plt.cm.tab10(5)  ]
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
            if remove_code:
                    removed_idx = len(y_vals)//2
                    x_vals = np.delete(x_vals, removed_idx)
                    y_vals = np.delete(y_vals, removed_idx)
                    print(f"Removed code at index {removed_idx} for column {col + 1}")

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
            maybe_suptitle(ax_dnl[corner], f"{clean_title} - DNL ({corner})", fontsize=14, fontweight='bold')
            ax_dnl[corner].set_xlabel("Digital Code", fontsize=12, fontweight='bold')
            ax_dnl[corner].set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
            self._apply_grid_styling(ax_dnl[corner], alpha=0.35)
            ax_dnl[corner].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
            ax_dnl[corner].set_ylim(-1, 1)
            ax_dnl[corner].legend(bbox_to_anchor=(1.05, 1), fontsize=6, framealpha=0.95, loc='best')

            # Format INL subplot (bottom)
            maybe_suptitle(ax_inl[corner], f"{clean_title} - INL ({corner})", fontsize=14, fontweight='bold')
            ax_inl[corner].set_xlabel("Digital Code", fontsize=12, fontweight='bold')
            ax_inl[corner].set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
            self._apply_grid_styling(ax_inl[corner], alpha=0.35)
            ax_inl[corner].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
            # ax_inl[corner].legend(fontsize=6, framealpha=0.95, loc='best')

            plt.tight_layout()
            save_path = self._get_save_path(filename, "Linearity", f"corner_dnl_inl_{corner.lower()}")
            self._save_figure(fig[corner], save_path)
            
            # Format and save sweep plot
            maybe_suptitle(ax_sweep[corner], f"Sweep Plot - Code vs Output Delay ({corner})", fontsize=14, fontweight='bold')
            ax_sweep[corner].set_xlabel("Digital Code", fontsize=12, fontweight='bold')
            ax_sweep[corner].set_ylabel("Output Delay (s)", fontsize=12, fontweight='bold')
            self._apply_grid_styling(ax_sweep[corner], alpha=0.35)
            ax_sweep[corner].legend(bbox_to_anchor=(1.05, 1), fontsize=6, framealpha=0.95, loc='best')

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
        print(f"Found X columns: {x_cols}, Y columns: {y_cols} in {filename}")
        if not x_cols or not y_cols: return None
        
        fig, ax = plt.subplots()
        ax.plot(df[x_cols[0]], df[y_cols[0]]*1e12, marker=None, 
               linewidth=2.6)
        
        title = self._format_title(path.stem)
        maybe_title(ax, f"Sweep: {title}"
        , fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(r"Delay [ps]", fontsize=12, fontweight='bold')
        ax.set_xlabel(r"$V_{dd} \, [V]$", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"generic_sweep")
        self._save_figure(fig, save_path)
        print(f"Saved generic sweep plot to {save_path}")
        plt.close()

    def _draw_bus_lane(self, ax, x, y_labels, color, t_min, t_max):
        """
        Draws two continuous parallel lines across the entire signal window,
        sharp vertical divider walls at transition steps, and clean unsigned
        decimal integers positioned directly in the middle.
        """
        # Set explicitly 2 ticks for the digital code bus lanes: [0, 1]
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.2, 1.2)
        
        total_span = t_max - t_min

        # 1. Draw the two horizontal bounding lines for the entire overall simulation duration
        ax.plot([t_min, t_max], [1, 1], color=color, linewidth=1.5)
        ax.plot([t_min, t_max], [0, 0], color=color, linewidth=1.5)
        
        # 2. Extract transitions safely
        unique_x = [t_min]
        unique_vals = [y_labels[0]]
        
        for i in range(1, len(x)):
            if y_labels[i] != y_labels[i-1]:
                unique_x.append(x[i-1])
                unique_vals.append(y_labels[i-1])
                unique_x.append(x[i])
                unique_vals.append(y_labels[i])
                
        unique_x.append(t_max)
        unique_vals.append(y_labels[-1])
        
        intervals = []
        curr_start = unique_x[0]
        curr_val = unique_vals[0]
        
        for i in range(1, len(unique_x)):
                intervals.append((curr_start, unique_x[i], curr_val))
                curr_start = unique_x[i]
                curr_val = unique_vals[i]
        intervals.append((curr_start, unique_x[-1], curr_val))

        # 3. Draw vertical step walls and write clean decimal code labels inside
        for idx, (t_start, t_end, val) in enumerate(intervals):
            # Draw vertical line segment divider wall
            ax.plot([t_start, t_start], [0, 1], color=color, linewidth=1.5)
            if idx == len(intervals) :
                ax.plot([t_end, t_end], [0, 1], color=color, linewidth=1.5)

            # Convert binary code string to unsigned decimal integer
            try:
                clean_str = str(val).strip()
                if re.match(r'^[01]+$', clean_str):
                    dec_val = int(clean_str, 2)
                else:
                    dec_val = int(float(clean_str))
            except ValueError:
                dec_val = val
                
            # Print text perfectly centered at height 0.5
            # Bypassed width constraint for the last interval so it always renders
            if (t_end - t_start) > (total_span * 0.01):
                t_mid = (t_start + t_end) / 2
                ax.text(t_mid, 0.5, str(dec_val), color='black', va='center', ha='center', 
                        fontsize=15, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

    def plot_signals(self, df=None, subplots=True, file_a=None, file_b=None, **kwargs):
        """
        Plots and overlays analog signal waveforms from Pre-layout (--file_a) 
        and Post-layout (--file_b) files grouped by signal name.
        File A (Pre) uses a dashed line, File B (Post) uses a solid line.
        Digital code buses are kept and plotted exactly once using a rectangular box layout.
        """
        df_a = None
        df_b = None
        
        # Load File A and File B dataframes explicitly
        if file_a:
            df_a, _ = self.load_data(file_a)
        if file_b:
            df_b, _ = self.load_data(file_b)
            
        # Fallback to single df argument if separate files are not provided
        if df_a is None and df_b is None:
            if isinstance(df, (str, Path)):
                df_b, _ = self.load_data(df)
            else:
                df_b = df

        # Helper lambda to parse signal columns from a dataframe
        def extract_signal_pairs(dataframe):
            if dataframe is None:
                return {}
            pairs = {}
            columns = dataframe.columns
            for i in range(0, len(columns), 2):
                if i + 1 < len(columns):
                    x_col = columns[i]
                    y_col = columns[i+1]
                    clean_name = y_col.replace('/', '').replace(' Y', '').replace(' LogicBus', '').strip()
                    is_bus = "LogicBus" in y_col or clean_name in ['B', 'D']
                    pairs[clean_name] = (x_col, y_col, is_bus)
            return pairs

        pairs_a = extract_signal_pairs(df_a)
        pairs_b = extract_signal_pairs(df_b)
        
        # Get the complete unique sorted list of all signals
        all_signal_names = list(set(pairs_a.keys()) | set(pairs_b.keys()))
        print(f"Identified signal names: {all_signal_names}")
        num_signals = len(all_signal_names)
        
        if num_signals == 0:
            print("No matching signals found to plot.")
            return

        # Synchronize timeline boundaries globally across all valid tracks
        g_min, g_max = float('inf'), float('-inf')
        
        for p in list(pairs_a.values()) + list(pairs_b.values()):
            df_target = df_a if p in pairs_a.values() else df_b
            x_col = p[0]
            times = pd.to_numeric(df_target[x_col], errors='coerce').dropna().values
            if len(times) > 0:
                g_min, g_max = min(g_min, times[0]), max(g_max, times[-1])

        if subplots:
            fig, axes = plt.subplots(
                nrows=num_signals-1, 
                ncols=1, 
                figsize=(7, 1 *(num_signals-1)), 
                sharex=True, 
                constrained_layout=True
            )
            if num_signals == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            axes = [ax] * num_signals

        # High-contrast color map to assign a distinct uniform color per signal row
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_signals)))

        all_signal_names = ['in_DTC', 'out_DTC','V_ramp' , 'D' ]#['in_DTC','EN_b', 'out_DTC', 'V_cap','V_ramp' , 'B', 'D' ]
        for idx, name in enumerate(all_signal_names):
            ax = axes[idx] if subplots else axes[0]
            curve_color = colors[idx]
            
            # Determine if this signal row is a digital bus track
            is_bus = (name in pairs_b and pairs_b[name][2]) or (name in pairs_a and pairs_a[name][2])
            
            if is_bus:
                # Digital buses: Plot ONLY ONCE using the active layout file
                target_pairs = pairs_b if name in pairs_b else pairs_a
                target_df = df_b if name in pairs_b else df_a
                
                x_col, y_col, _ = target_pairs[name]
                data_bus = target_df[[x_col, y_col]].dropna()
                x_val = pd.to_numeric(data_bus[x_col], errors='coerce').values
                bus_labels = data_bus[y_col].astype(str).str.strip().values
                
                self._draw_bus_lane(ax, x_val, bus_labels, curve_color, g_min, g_max)
                ax.grid(True, axis='x', linestyle='--', alpha=0.4)
            else:
                # Analog signals: Overlay Pre (dashed) and Post (solid)
                y_data_points = []

                if name in pairs_a:
                    x_col, y_col, _ = pairs_a[name]
                    data_a = df_a[[x_col, y_col]].dropna()
                    x_val = pd.to_numeric(data_a[x_col], errors='coerce').values
                    y_val = pd.to_numeric(data_a[y_col], errors='coerce').values
                    ax.plot(x_val, y_val, color=curve_color, linestyle='--', linewidth=2, label='Pre')
                    y_data_points.extend(y_val)
                    

                if name in pairs_b:
                    x_col, y_col, _ = pairs_b[name]
                    data_b = df_b[[x_col, y_col]].dropna()
                    x_val = pd.to_numeric(data_b[x_col], errors='coerce').values
                    y_val = pd.to_numeric(data_b[y_col], errors='coerce').values
                    ax.plot(x_val, y_val, color=curve_color, linestyle='-', linewidth=2, label='Post')
                    y_data_points.extend(y_val)

                # Set custom vertical scales and ticks
                if "V_n" in name :
                        max_y = np.nanmax(y_data_points) if len(y_data_points) > 0 else 0
                        ymin, ymax = (500, 700) if max_y > 10 else (0.5, 0.7)
                        ax.set_ylim(ymin, ymax)
                        ax.set_yticks(np.linspace(ymin, ymax, 3))
                elif "V_cap" in name:
                        max_y = np.nanmax(y_data_points) if len(y_data_points) > 0 else 0
                        ymin, ymax = (500, 700) if max_y > 10 else (0.7, 1.5)
                        ax.set_ylim(ymin, ymax)
                        ax.set_yticks(np.linspace(0, ymax, 3))

                elif "V_ramp" in name:
                        max_y = np.nanmax(y_data_points) if len(y_data_points) > 0 else 0
                        ymin, ymax = (500, 700) if max_y > 10 else (-0.1, 1.2) #(-0.1, 1.5)
                        ax.set_ylim(ymin, ymax)
                        ax.set_yticks(np.linspace(0, 1.1, 3))
                else:
                        # Regular signals get exactly 0, 0.55, and 1.1
                        ax.set_ylim(-0.1, 1.2)
                        ax.set_yticks([0, 0.55, 1.1])
                
                ax.grid(True, linestyle='--', alpha=0.4)

            # Ensure timeline doesn't truncate early on the right edge
            ax.set_xlim(g_min, g_max)

            # LaTeX Style Formatting (e.g. converting out_DTC -> out_{\text{DTC}})
            if subplots:
                if "_" in name:
                    base, subscript = name.split("_", 1)
                    latex_name = f"${base}_{{{{\\text{{{subscript}}}}}}}$"
                else:
                    latex_name = name
                ax.set_ylabel(latex_name, rotation=0, labelpad=40, va='center', fontweight='bold')
            
            # Place a small legend tracking pre/post styling helper profiles on row 0
            # Place the sharp black border legend on the first non-bus trace block row
            if idx == 0 and not is_bus:
                ax.legend(
                    loc='upper right', 
                    fontsize=12, 
                    framealpha=1.0, 
                    edgecolor='black', 
                    facecolor='white',
                    labelcolor='black'
)

        if subplots:
            axes[-1].set_xlabel("Time (s)")
        else:
            axes[0].set_xlabel("Time (s)")

        out_path = self.plot_dir / "signals_comparison_plot.pdf"
        plt.savefig(out_path, dpi=300)
        print(f"Saved compared signal layout plots to: {out_path}")
        plt.show()
    
       

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
                fig, axes = plt.subplots(
                    len(unique_corners),
                    1,
                    figsize=_multi_panel_figsize(len(unique_corners), 1),
                    sharex=True,
                )
                if len(unique_corners) == 1: axes = [axes]
            else:
                fig = plt.figure()
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
                    maybe_title(ax, f"Corner: {corner_label}", fontsize=12, fontweight='bold', pad=10)
                ax.set_ylabel(f"{y_label} ({unit})", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, framealpha=0.95)
                ax.set_axisbelow(True)

            if not subfigure:
                maybe_suptitle(axes[0], f"{y_label} vs Digital Code", fontsize=14, fontweight='bold', pad=15)
            
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
            fig, axes = plt.subplots(
                num_rows,
                2,
                figsize=_multi_panel_figsize(num_rows, 2),
                sharex=True,
            )
            # Ensure axes is 2D even if only one corner exists
            if num_rows == 1: axes = np.expand_dims(axes, axis=0)
        else:
            fig, (ax_dnl, ax_inl) = plt.subplots(
                2,
                1,
                figsize=_multi_panel_figsize(2, 1),
                sharex=True,
            )
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
                maybe_suptitle(cur_ax_dnl, f"DNL - {corner_label}", fontsize=12, fontweight='bold', pad=10)
                maybe_suptitle(cur_ax_inl, f"INL - {corner_label}", fontsize=12, fontweight='bold', pad=10)
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
            maybe_suptitle(axes[0][0], "Differential Non-Linearity (DNL)", fontsize=14, fontweight='bold', pad=15)
            maybe_suptitle(axes[0][1], "Integral Non-Linearity (INL)", fontsize=14, fontweight='bold', pad=15)
            axes[0][0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, framealpha=0.95)
            
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
            "phase_noise": "Phase noise comparison (CS vs VS)",
            "pvt_sweep": "PVT sweep",
            "pvt_linearity": "PVT linearity",
            "pvt_envelopes": "PVT DNL/INL envelope",
            "pvt_summary": "PVT summary plots",
        }
    
    def plot_linearity_envelope(self, file_a=None, file_b=None, num_codes=256, **kwargs):
        """
        Plots the DNL and INL performance bounds over all corners using an envelope/spread plot.
        Compares Pre-layout (--file_a) vs Post-layout (--file_b).
        Pre-layout uses a dashed mean line with a light blue shade.
        Post-layout uses a solid mean line with an orange shade.
        """
        df_a = None
        df_b = None
        
        if file_a:
            df_a, _ = self.load_data(file_a)
        if file_b:
            df_b, _ = self.load_data(file_b)

        if df_a is None and df_b is None:
            print("Error: Please provide both --file_a and --file_b for envelope comparison.")
            return

        # Dynamically determine the actual number of codes available in the files
        sample_df = df_b if df_b is not None else df_a
        actual_num_codes = num_codes
        if sample_df is not None and len(sample_df.columns) > 1:
            # Check length of the first valid data column to set the true size loop bound
            actual_num_codes = len(sample_df.iloc[:, 1].dropna().values)
            if actual_num_codes > num_codes:
                actual_num_codes = num_codes

        def calculate_linearity_matrix(df):
            if df is None:
                return None, None
            
            columns = df.columns
            all_dnl = []
            all_inl = []
            
            # Walk through every column pair (each representing one PVT corner)
            for i in range(0, len(columns), 2):
                if i + 1 < len(columns):
                    x_col = columns[i]
                    y_col = columns[i+1]
                    data = df[[x_col, y_col]].dropna()
                    y_val = pd.to_numeric(data[y_col], errors='coerce').values[:actual_num_codes]
                    
                    if len(y_val) < 2:
                        continue
                    
                    # Ideal LSB step calculation
                    lsb_ideal = (y_val[-1] - y_val[0]) / (len(y_val) - 1)
                    
                    if lsb_ideal != 0:
                        # DNL = (Actual Step / Ideal Step) - 1
                        steps = np.diff(y_val)
                        dnl = (steps / lsb_ideal) - 1.0
                        # Pad DNL with a trailing 0 to keep array dimension identical to codes exactly
                        dnl = np.append(dnl, 0.0)
                        
                        # INL = (Actual Value - Ideal Straight Line) / Ideal Step
                        ideal_line = np.linspace(y_val[0], y_val[-1], len(y_val))
                        inl = (y_val - ideal_line) / lsb_ideal
                    else:
                        dnl = np.zeros(len(y_val))
                        inl = np.zeros(len(y_val))
                        
                    all_dnl.append(dnl[:actual_num_codes])
                    all_inl.append(inl[:actual_num_codes])
            
            return np.array(all_dnl), np.array(all_inl)

        # Calculate linearity matrices safely bound to actual_num_codes
        dnl_matrix_a, inl_matrix_a = calculate_linearity_matrix(df_a)
        dnl_matrix_b, inl_matrix_b = calculate_linearity_matrix(df_b)

        # Define 2-panel layout
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, constrained_layout=True)
        
        plot_codes = np.arange(actual_num_codes)
        color_pre = plt.cm.tab10(0)   # Blue
        color_post = plt.cm.tab10(1)  # Orange

        def draw_envelope_layer(ax, matrix, color, is_post=False):
            if matrix is None or len(matrix) == 0:
                return
            mean_val = np.mean(matrix, axis=0)
            min_val = np.min(matrix, axis=0)
            max_val = np.max(matrix, axis=0)
            
            line_style = '-' if is_post else '--'
            label_str = 'Post-layout Bounds' if is_post else 'Pre-layout Bounds'
            
            # Plot the central mean line
            ax.plot(plot_codes, mean_val, color=color, linestyle=line_style, linewidth=2.0, label=label_str)
            # Fill the spread envelope area
            ax.fill_between(plot_codes, min_val, max_val, color=color, alpha=0.2 if is_post else 0.15)

        # --- Panel 1: DNL Envelope Shading ---
        draw_envelope_layer(ax1, dnl_matrix_a, color=color_pre, is_post=False)
        draw_envelope_layer(ax1, dnl_matrix_b, color=color_post, is_post=True)
        ax1.set_ylabel("Differential Non-Linearity ($DNL$, LSB)", fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.35)

        # --- Panel 2: INL Envelope Shading ---
        draw_envelope_layer(ax2, inl_matrix_a, color=color_pre, is_post=False)
        draw_envelope_layer(ax2, inl_matrix_b, color=color_post, is_post=True)
        ax2.set_ylabel("Integral Non-Linearity ($INL$, LSB)", fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.35)

        # Common X Axis Settings
        ax2.set_xlabel("Digital Input Code", fontsize=11, fontweight='bold')
        ax2.set_xlim(0, actual_num_codes - 1)

        # Apply clean black-bordered font-15 legend configuration
        ax1.legend(
            loc='upper right', 
            fontsize=15, 
            framealpha=1.0, 
            edgecolor='black', 
            facecolor='white',
            labelcolor='black'
        )

        fig.suptitle("PVT Corner Variation Envelope: Linearity Error Comparison ($DNL$ / $INL$)", fontsize=13, fontweight='bold')

        out_path = self.plot_dir / "linearity_envelope_comparison.pdf"
        plt.savefig(out_path, dpi=300)
        print(f"Saved clean variation envelope plot to: {out_path}")
        plt.show()

    def plot_file(self, filename, plot_type="auto", **kwargs):
        """Single entry point for plotting a file by explicit plot type."""
        plot_type = (plot_type or "auto").lower().strip()

        print(f"Plotting '{filename}' with plot type '{plot_type}'...")

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
        print(kwargs.get("remove_code"))
        dispatch = {
            "auto": lambda: self.smart_plot(filename),
            "sweep": lambda: self.plot_digital_sweep(filename, file_a = kwargs.get("file_a"),
                file_b = kwargs.get("file_b"), max_iterations=kwargs.get("max_iterations", 200)),
            "linearity": lambda: self.plot_linearity(filename),
            "signals": lambda: self.plot_signals(
                filename,
                filters=kwargs.get("filters"),
                t_range=kwargs.get("t_range"),
                subplots=kwargs.get("subplots", False),
                file_a = kwargs.get("file_a"),
                file_b = kwargs.get("file_b"),
            ),
            "pvt_grid": lambda: self.plot_pvt_linearity_grid(file_a=kwargs.get("file_a"), file_b=kwargs.get("file_b"), num_codes=kwargs.get("num_codes", 256)),
            "pvt_dnl_inl": lambda: self.plot_peak_linearity_errors(file_a=kwargs.get("file_a"), file_b=kwargs.get("file_b"), num_codes=kwargs.get("num_codes", 256)),
            "pvt_envelopes": lambda: self.plot_linearity_envelope(file_a=kwargs.get("file_a"), file_b=kwargs.get("file_b"), num_codes=kwargs.get("num_codes", 256)),
            "pvt_recap": lambda: self.plot_metric_summary(file_a=kwargs.get("file_a"), file_b=kwargs.get("file_b"), num_codes=kwargs.get("num_codes", 256)),
            "histogram": lambda: self.plot_histogram(filename),
            "mc_sweep": lambda: self.plot_mc_sweep_all_realizations(filename, max_realizations=kwargs.get("max_realizations", 200), remove_code=kwargs.get("remove_code", True)),
            "mc_linearity": lambda: self.plot_mc_linearity_all_realizations_xy(filename, max_realizations=kwargs.get("max_realizations", 200), remove_code=kwargs.get("remove_code", True)),
            "mc_linearity_iteration": lambda: self.plot_mc_linearity_per_iteration(
                filename,
                lsb_ns=kwargs.get("lsb_ns"),
                max_iterations=kwargs.get("max_iterations", 200), remove_code=kwargs.get("remove_code", True)
            ),
            "mcparamset_sweep": lambda: self.plot_mcparamset_sweep(filename, max_realizations=kwargs.get("max_realizations", 200), remove_code=kwargs.get("remove_code", True)),
            "mcparamset_linearity": lambda: self.plot_mcparamset_linearity(filename, max_realizations=kwargs.get("max_realizations", 200), remove_code=kwargs.get("remove_code", True)),
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
                file_a = kwargs.get("file_a"),
                file_b = kwargs.get("file_b"),
            ),
            "corner_temp": lambda: self.plot_corner_temperature_sweep(filename),
            "corner_linearity": lambda: self.plot_corner_linearity_xy(filename, remove_code=kwargs.get("remove_code", True)),
            "generic": lambda: self.plot_generic_sweep(filename),
            "phase_noise": lambda: self.plot_phase_noise_comparison(
                kwargs.get("file_a") or kwargs.get("cs_file"),
                kwargs.get("file_b") or kwargs.get("vs_file"),
                title=kwargs.get("phase_noise_title"),
                log_x=not kwargs.get("no_log_x", False),
                carrier_hz=kwargs.get("carrier_hz"),
                label_a=kwargs.get("label_a"),
                label_b=kwargs.get("label_b"),
            ),
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
    
    def plot_metric_summary(self, file_a=None, file_b=None, num_codes=256, **kwargs):
        """
        Computes summary metrics (Peak DNL, Peak INL, Dynamic Range, and Resolution) comparing 
        Pre-layout (--file_a) vs Post-layout (--file_b) across all corners.
        Exports the clean grouped metrics directly into a CSV spreadsheet.
        """
        df_a = None
        df_b = None
        
        if file_a:
            df_a, _ = self.load_data(file_a)
        if file_b:
            df_b, _ = self.load_data(file_b)

        if df_a is None and df_b is None:
            print("Error: Please provide both --file_a and --file_b for summary extraction.")
            return

        def extract_corners(df):
            if df is None:
                return {}
            corners = {}
            columns = df.columns
            for i in range(0, len(columns), 2):
                if i + 1 < len(columns):
                    x_col = columns[i]
                    y_col = columns[i+1]
                    clean_name = y_col.replace('delay (', '').replace(') Y', '').replace(') X', '').strip()
                    clean_name = clean_name.replace('modelFiles=toplevel.scs:', '')
                    corners[clean_name] = (x_col, y_col)
            return corners

        corners_a = extract_corners(df_a)
        corners_b = extract_corners(df_b)
        all_corners = sorted(list(set(corners_a.keys()) | set(corners_b.keys())))

        # Internal helper to calculate linearity parameters
        def compute_metrics(df, corner_dict, corner_name):
            if corner_name not in corner_dict:
                return None, None, None, None
            x_col, y_col = corner_dict[corner_name]
            data = df[[x_col, y_col]].dropna()
            y_val = pd.to_numeric(data[y_col], errors='coerce').values[:num_codes]
            
            if len(y_val) < 2:
                return 0.0, 0.0, 0.0, 0.0
                
            # 1. Dynamic Range (DR) in ns
            dr = (np.max(y_val) - np.min(y_val)) * 1e9
            
            # 2. Resolution (Average LSB Step Size) in ps
            res = (dr * 1e3) / (len(y_val) - 1)
            
            # 3. Linearity calculations (DNL and INL)
            lsb_ideal = (y_val[-1] - y_val[0]) / (len(y_val) - 1)
            if lsb_ideal != 0:
                # DNL = (Actual Step / Ideal Step) - 1
                steps = np.diff(y_val)
                dnl = (steps / lsb_ideal) - 1.0
                peak_dnl = np.max(np.abs(dnl))
                
                # INL = (Actual Value - Ideal Straight Line) / Ideal Step
                ideal_line = np.linspace(y_val[0], y_val[-1], len(y_val))
                inl = (y_val - ideal_line) / lsb_ideal
                peak_inl = np.max(np.abs(inl))
            else:
                peak_dnl = 0.0
                peak_inl = 0.0
                
            return peak_dnl, peak_inl, dr, res

        # Collect metrics into row structures
        rows = []
        for corner in all_corners:
            p_dnl_a, p_inl_a, dr_a, res_a = compute_metrics(df_a, corners_a, corner)
            p_dnl_b, p_inl_b, dr_b, res_b = compute_metrics(df_b, corners_b, corner)
            
            rows.append({
                'PVT_Corner': corner,
                'Pre_Peak_DNL_LSB': round(p_dnl_a, 3) if p_dnl_a is not None else 0.0,
                'Post_Peak_DNL_LSB': round(p_dnl_b, 3) if p_dnl_b is not None else 0.0,
                'Pre_Peak_INL_LSB': round(p_inl_a, 3) if p_inl_a is not None else 0.0,
                'Post_Peak_INL_LSB': round(p_inl_b, 3) if p_inl_b is not None else 0.0,
                'Pre_Dynamic_Range_ns': round(dr_a, 3) if dr_a is not None else 0.0,
                'Post_Dynamic_Range_ns': round(dr_b, 3) if dr_b is not None else 0.0,
                'Pre_LSB_Resolution_ps': round(res_a, 3) if res_a is not None else 0.0,
                'Post_LSB_Resolution_ps': round(res_b, 3) if res_b is not None else 0.0
            })

        # Convert row dictionaries to a pandas DataFrame
        summary_df = pd.DataFrame(rows)
        
        # Define the target directory path and export filename
        out_path = self.plot_dir / "performance_metric_summary.csv"
        summary_df.to_csv(out_path, index=False)
        
        print("\n" + "="*85)
        print(f" SUCCESS: Exported complete summary metrics to CSV!")
        print(f" Target Path: {out_path}")
        print("="*85)
        print(summary_df.to_string(index=False))
        print("="*85 + "\n")
        
        return out_path
    
    def plot_peak_linearity_errors(self, file_a=None, file_b=None, num_codes=32, **kwargs):
        """
        Plots the peak absolute values of DNL and INL on two separated line plots.
        X-axis shows ordered Process and Temperature configurations (e.g., SS, -55).
        Lines change color based on Vdd and connect the points together.
        Relies on parent frame configurations for figure sizes, labels, and legends.
        """
        df_a, _ = self.load_data(file_a) if file_a else (None, None)
        df_b, _ = self.load_data(file_b) if file_b else (None, None)
        df_target = df_b if df_b is not None else df_a
        
        if df_target is None:
            print("Error: No valid data available.")
            return

        def parse_header(col_name):
            clean = col_name.replace('delay (modelFiles=toplevel.scs:', '').replace(') Y', '').strip()
            match = re.search(r'([^,]+),Vdd=([^,]+),temperature=([^)]+)', clean)
            if match:
                proc = match.group(1).replace('top_', '').upper()  # Strips 'top_' -> 'SS', 'TT'
                return proc, match.group(2), match.group(3)
            return None

        data_points_a = {}
        data_points_b = {}
        found_procs = set()
        found_temps = set()

        columns = df_target.columns
        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                parsed = parse_header(columns[i+1])
                if not parsed:
                    continue
                proc, vdd, temp = parsed
                cond_key = f"{proc},{temp}"
                
                found_procs.add(proc)
                found_temps.add(temp)

                def get_peaks(df):
                    if df is not None and columns[i+1] in df.columns:
                        y_val = pd.to_numeric(df[columns[i+1]], errors='coerce').dropna().values[:num_codes]
                        if len(y_val) >= 2:
                            lsb = (y_val[-1] - y_val[0]) / (len(y_val) - 1)
                            if lsb != 0:
                                raw_steps = np.diff(y_val)
                                dnl_steps = (raw_steps / lsb) - 1.0
                                peak_dnl = np.max(np.abs(dnl_steps))
                                
                                ideal_line = np.linspace(y_val[0], y_val[-1], len(y_val))
                                inl_steps = (y_val - ideal_line) / lsb
                                peak_inl = np.max(np.abs(inl_steps))
                                
                                return peak_dnl, peak_inl
                    return None

                peaks_a = get_peaks(df_a)
                if peaks_a:
                    if vdd not in data_points_a: data_points_a[vdd] = {}
                    data_points_a[vdd][cond_key] = peaks_a

                peaks_b = get_peaks(df_b)
                if peaks_b:
                    if vdd not in data_points_b: data_points_b[vdd] = {}
                    data_points_b[vdd][cond_key] = peaks_b

        sorted_procs = sorted(list(found_procs))
        
        def temp_sort_key(t_str):
            try:
                return float(t_str)
            except ValueError:
                return 999.0
        sorted_temps = sorted(list(found_temps), key=temp_sort_key)

        all_conditions = []
        for p in sorted_procs:
            for t in sorted_temps:
                cond_str = f"{p},{t}"
                if any(cond_str in d for d in [data_points_a.get(v, {}) for v in data_points_a] + [data_points_b.get(v, {}) for v in data_points_b]):
                    all_conditions.append(cond_str)

        x_indices = np.arange(len(all_conditions))
        vdd_colors = {'0.88': plt.cm.tab10(0), '1.1': plt.cm.tab10(1), '1.32': plt.cm.tab10(2)}

        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([0], [0], color='black', linestyle='-', label='Post-Layout'),
            Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Pre-Layout')
        ]

        # --- Figure 1: Separated Peak DNL Trend ---
        # Requesting a slightly expanded widescreen layout from your figure generator
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        for vdd, cond_dict in data_points_a.items():
            color = vdd_colors.get(vdd, 'black')
            y_pts = [cond_dict[c][0] if c in cond_dict else np.nan for c in all_conditions]
            ax1.plot(x_indices, y_pts, color=color, linestyle='--', marker='o', alpha=0.4)

        for vdd, cond_dict in data_points_b.items():
            color = vdd_colors.get(vdd, 'black')
            y_pts = [cond_dict[c][0] if c in cond_dict else np.nan for c in all_conditions]
            ax1.plot(x_indices, y_pts, color=color, linestyle='-', marker='o', label=rf'$Vdd={vdd}$ V')

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(all_conditions, rotation=35, ha='right')
        
       
        ax1.set_xlabel("PVT Operating Corners")
        ax1.set_ylabel("Peak |$DNL$| (LSB)")
            
        if hasattr(self, '_apply_grid_styling'):
            self._apply_grid_styling(ax1, alpha=0.35)
        else:
            ax1.grid(True, linestyle='--', alpha=0.4)
        
        current_handles, current_labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=current_handles + style_handles, loc='best')
        
        plt.tight_layout()
        out_dnl = self.plot_dir / "peak_dnl_trends_comparison.pdf"
        plt.savefig(out_dnl, dpi=300)
        plt.show()

        # --- Figure 2: Separated Peak INL Trend ---
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        for vdd, cond_dict in data_points_a.items():
            color = vdd_colors.get(vdd, 'black')
            y_pts = [cond_dict[c][1] if c in cond_dict else np.nan for c in all_conditions]
            ax2.plot(x_indices, y_pts, color=color, linestyle='--', marker='s', alpha=0.4)

        for vdd, cond_dict in data_points_b.items():
            color = vdd_colors.get(vdd, 'black')
            y_pts = [cond_dict[c][1] if c in cond_dict else np.nan for c in all_conditions]
            ax2.plot(x_indices, y_pts, color=color, linestyle='-', marker='s', label=rf'$Vdd={vdd}$ V')

        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(all_conditions, rotation=35, ha='right')
        
        if hasattr(self, '_format_plot_labels'):
            self._format_plot_labels(ax2, xlabel="PVT Operating Corners", ylabel="Peak |$INL$| (LSB)", title="Peak INL Variation Trends")
        else:
            ax2.set_xlabel("PVT Operating Corners")
            ax2.set_ylabel("Peak |$INL$| (LSB)")
            
        if hasattr(self, '_apply_grid_styling'):
            self._apply_grid_styling(ax2, alpha=0.35)
        else:
            ax2.grid(True, linestyle='--', alpha=0.4)
            
        current_handles2, current_labels2 = ax2.get_legend_handles_labels()
        ax2.legend( handles=current_handles + style_handles,loc='best')
        
        plt.tight_layout()
        out_inl = self.plot_dir / "peak_inl_trends_comparison.pdf"
        plt.savefig(out_inl, dpi=300)
        plt.show()
        
        print(f"Saved trend charts to:\n  - {out_dnl}\n  - {out_inl}")

    def plot_pvt_linearity_grid(self, file_a=None, file_b=None, num_codes=32, **kwargs):
        """
        Plots DNL and INL across 5 subplots (one per Process Corner).
        Color maps to Vdd, Line Style maps to Temperature.
        Compares Pre-layout (thinner, alpha) vs Post-layout (bolder).
        """
        df_a, _ = self.load_data(file_a) if file_a else (None, None)
        df_b, _ = self.load_data(file_b) if file_b else (None, None)
        df_target = df_b if df_b is not None else df_a
        
        if df_target is None:
            print("Error: No valid data available.")
            return

        def parse_header(col_name):
            # Regex to clean up and extract PVT values from Cadence syntax
            clean = col_name.replace('delay (modelFiles=toplevel.scs:', '').replace(') Y', '').strip()
            match = re.search(r'([^,]+),Vdd=([^,]+),temperature=([^)]+)', clean)
            if match:
                return match.group(1), match.group(2), match.group(3)
            return "unknown", "unknown", "unknown"

        # Organize all available columns into a nested dictionary structure
        data_tree = {}
        columns = df_target.columns
        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                proc, vdd, temp = parse_header(columns[i+1])
                if proc not in data_tree: data_tree[proc] = []
                data_tree[proc].append((vdd, temp, columns[i], columns[i+1]))

        proc_corners = sorted(list(data_tree.keys()))
        num_plots = len(proc_corners) if proc_corners else 1

        fig, axes = plt.subplots(nrows=num_plots, ncols=2, figsize=(16, 2.5 * num_plots), sharex=True, constrained_layout=True)
        if num_plots == 1: axes = np.expand_dims(axes, axis=0)

        # Style maps for consistent visual grouping
        vdd_colors = {'0.88': plt.cm.tab10(0), '1.1': plt.cm.tab10(1), '1.32': plt.cm.tab10(2)}
        temp_styles = {'-55': ':', '27': '--', '125': '-'}

        for p_idx, proc in enumerate(proc_corners):
            ax_dnl, ax_inl = axes[p_idx, 0], axes[p_idx, 1]
            
            for vdd, temp, x_col, y_col in data_tree[proc]:
                color = vdd_colors.get(vdd, 'black')
                style = temp_styles.get(temp, '-')
                
                # Process Pre-layout if present
                if df_a is not None and y_col in df_a.columns:
                    y_val = pd.to_numeric(df_a[y_col], errors='coerce').dropna().values[:num_codes]
                    if len(y_val) >= 2:
                        lsb = (y_val[-1] - y_val[0]) / (len(y_val) - 1)
                        dnl = np.append((np.diff(y_val) / lsb) - 1.0, 0.0)
                        inl = (y_val - np.linspace(y_val[0], y_val[-1], len(y_val))) / lsb
                        ax_dnl.plot(dnl, color=color, linestyle=style, linewidth=1.0, alpha=0.4)
                        ax_inl.plot(inl, color=color, linestyle=style, linewidth=1.0, alpha=0.4)

                # Process Post-layout if present
                if df_b is not None and y_col in df_b.columns:
                    y_val = pd.to_numeric(df_b[y_col], errors='coerce').dropna().values[:num_codes]
                    if len(y_val) >= 2:
                        lsb = (y_val[-1] - y_val[0]) / (len(y_val) - 1)
                        dnl = np.append((np.diff(y_val) / lsb) - 1.0, 0.0)
                        inl = (y_val - np.linspace(y_val[0], y_val[-1], len(y_val))) / lsb
                        ax_dnl.plot(dnl, color=color, linestyle=style, linewidth=2.2)
                        ax_inl.plot(inl, color=color, linestyle=style, linewidth=2.2)

            ax_dnl.grid(True, alpha=0.3)
            ax_inl.grid(True, alpha=0.3)
            ax_dnl.set_ylabel(f"${proc}$\n$DNL$ (LSB)", fontsize=10, fontweight='bold')
            ax_inl.set_ylabel("$INL$ (LSB)", fontsize=10, fontweight='bold')

        axes[-1, 0].set_xlabel("Digital Input Code", fontsize=11, fontweight='bold')
        axes[-1, 1].set_xlabel("Digital Input Code", fontsize=11, fontweight='bold')
        
        # Build clean custom proxy handles for unified legendary parameters
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='-', label='Post-Layout (Bold)'),
            Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Pre-Layout (Thin)'),
            Line2D([0], [0], marker='s', color='none', markerfacecolor=plt.cm.tab10(0), label='$V_{\\text{dd}}=0.88\\text{ V}$'),
            Line2D([0], [0], marker='s', color='none', markerfacecolor=plt.cm.tab10(1), label='$V_{\\text{dd}}=1.1\\text{ V}$'),
            Line2D([0], [0], marker='s', color='none', markerfacecolor=plt.cm.tab10(2), label='$V_{\\text{dd}}=1.32\\text{ V}$'),
            Line2D([0], [0], color='black', linestyle=':', label='$T=-55^\\circ\\text{C}$'),
            Line2D([0], [0], color='black', linestyle='--', label='$T=27^\\circ\\text{C}$'),
            Line2D([0], [0], color='black', linestyle='-', label='$T=125^\\circ\\text{C}$')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=10, edgecolor='black')
        
        out_path = self.plot_dir / "pvt_linearity_grid_comparison.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved complete grid comparison layout to: {out_path}")
        plt.show()
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
    parser.add_argument("--cs-file", help="CS phase noise CSV filename")
    parser.add_argument("--vs-file", help="VS phase noise CSV filename")
    parser.add_argument("--file-a", help="Phase noise file A (optional)")
    parser.add_argument("--file-b", help="Phase noise file B (optional)")
    parser.add_argument("--label-a", default=None, help="Legend label for file A")
    parser.add_argument("--label-b", default=None, help="Legend label for file B")
    parser.add_argument("--phase-noise-title", default=None, help="Optional title for phase noise comparison")
    parser.add_argument("--no-log-x", action="store_true", help="Use linear X axis for phase noise")
    parser.add_argument("--carrier-hz", type=float, default=None, help="Carrier frequency in Hz for jitter integration")

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
        cs_file=args.cs_file,
        vs_file=args.vs_file,
        file_a=args.file_a,
        file_b=args.file_b,
        label_a=args.label_a,
        label_b=args.label_b,
        phase_noise_title=args.phase_noise_title,
        no_log_x=args.no_log_x,
        carrier_hz=args.carrier_hz,
    )


# --- Usage ---
if __name__ == "__main__":
    main()