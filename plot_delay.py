import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path

# --- Presentation Style ---
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.figsize": (12, 7),
    "lines.linewidth": 2,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

class CadencePlotter:
    def __init__(self,base_dir="results_cadence",plot_dir=r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\\plots"):
        self.base_dir = Path(base_dir)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)

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

    def _reconstruct_digital_data(self, df, filename, signal_name, bit_count):
        """Generic reconstruction for any bit depth with PI thermometer support."""
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
        
        # --- Remove Code 16 ONLY if it is a constant slope (cs) file ---
        if is_cs:
            if 16 in plot_df['code'].values and 15 in plot_df['code'].values:
                y15 = plot_df.loc[plot_df['code'] == 15, 'y'].values[0]
                y16 = plot_df.loc[plot_df['code'] == 16, 'y'].values[0]
                if np.isclose(y15, y16, rtol=1e-3):
                    plot_df = plot_df[plot_df['code'] != 16].reset_index(drop=True)
        
        return plot_df

    def plot_digital_sweep(self, filename, signal_name=None):
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        if signal_name is None:
            signal_name = df.columns[0].split(' (')[0].split(' ')[0]


        plot_df = self._reconstruct_digital_data(df, filename, signal_name, bit_count)
        if plot_df is None: return

        # 1. Determine Units and Scaling
        is_power = "P" in signal_name or "power" in signal_name.lower()
        unit = "W" if is_power else "s"
        scale = 1e6 if unit == "W" else 1e9
        prefix = "u" if unit == "W" else "n"

        # 2. Select Y data and Y label based on signal type
        y = plot_df['y'].values
        if is_power:
            # Power: Plot raw values from file
            y_to_plot = y * scale
            y_axis_label = f"{signal_name} ({prefix}{unit})"
        else:
            # Delay: Plot relative distance from first code
            y_to_plot = (y - y[0]) * scale
            y_axis_label = f"Relative Delay ({prefix}{unit})"

        # 3. Metrics (Calculated using raw range)
        dr = y.max() - y.min()
        res = dr / (len(y) - 1) if len(y) > 1 else 0

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(plot_df['label'], y_to_plot, marker='o', color='crimson', 
                 label=f"DR: {dr*scale:.2f}{prefix}{unit}\nRes: {res*scale*1000:.1f}p{unit}")
        
        plt.title(f"{bit_count}-Bit Sweep: {signal_name}")
        plt.xlabel(f"Digital Code (LSB={bit_count-1}...0)")
        plt.ylabel(y_axis_label)
        plt.xticks(rotation=90, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_path = self._get_save_path(filename, signal_name, "sweep")
        plt.savefig(save_path)
        plt.close()

    def plot_linearity(self, filename, signal_name=None):
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        if signal_name is None:
            signal_name = df.columns[0].split(' (')[0].split(' ')[0]

        plot_df = self._reconstruct_digital_data(df, filename, signal_name, bit_count)
        if plot_df is None: return

        y = plot_df['y'].values
        lsb_ideal = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
        dnl = np.insert(np.diff(y) / lsb_ideal - 1, 0, 0)
        inl = np.cumsum(dnl)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        ax1.plot(plot_df['label'], dnl, marker='o', color='blue', markersize=4)
        ax1.set_ylabel("DNL (LSB)")
        ax1.set_title(f"{bit_count}-Bit Linearity: {signal_name} (LSB={lsb_ideal:.2e})")
        
        ax2.plot(plot_df['label'], inl, marker='o', color='crimson', markersize=4)
        ax2.set_ylabel("INL (LSB)")
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()

        save_path = self._get_save_path(filename, signal_name, "linearity")
        plt.savefig(save_path)

        plt.close()

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
            
            ax.set_title(f"Monte Carlo: {label}\n($\mu$={mean:.3e}, $\sigma$={std:.3e})")
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
        plt.xlabel("Temperature [$^{\circ}$C]"); plt.ylabel(path.stem.split('_')[0].capitalize())
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


# --- Usage ---
plotter = CadencePlotter(base_dir="results_cadence")

# 1. Automatic Plots (Only for sweeps)
files = [
    "cs_delay_code_4bit.csv", "cs_delay_code_5bit.csv", "cs_delay_code_5bit_nonidealcurs.csv", "cs_power_code_4bit.csv", "cs_power_code_5bit.csv", "cs_delay_code_5bit_coarse.csv", "cs_power_code_5bit_nonidealcurs.csv", "cs_power_code_5bit_coarse.csv",
    "cs_delay_mc_tt_00000.csv", "cs_power_mc_tt_00000.csv", "cs_delay_mc_tt_11111.csv", "cs_power_mc_tt_11111.csv",  "cs_delay_mc_tt_10000.csv", "cs_power_mc_tt_10000.csv"]
for f in files:
    plotter.smart_plot(f)

# 2. plot transients
plotter.plot_signals("cs_Pinst_code_4bit.csv", filters=["d0=0,d1=0,d2=0,d3=0", "d0=0,d1=0,d2=0,d3=1",  "d0=1,d1=1,d2=1,d3=0", "d0=1,d1=1,d2=1,d3=1"], t_range=(40e-9,100e-9), subplots=True)
plotter.plot_signals("cs_signals_code_4bit.csv", filters=["d0=0,d1=0,d2=0,d3=0", "d0=0,d1=0,d2=0,d3=1", "d0=1,d1=1,d2=1,d3=0", "d0=1,d1=1,d2=1,d3=1"], t_range=(40e-9,100e-9), subplots=True)
plotter.plot_signals("cs_Pinst_code_5bit.csv", filters=["d0=0,d1=0,d2=0,d3=0,d4=0", "d0=0,d1=0,d2=0,d3=1,d4=1", "d0=0,d1=0,d2=0,d3=0,d4=1", "d0=1,d1=1,d2=1,d3=0,d4=0", "d0=1,d1=1,d2=1,d3=1,d4=0"], t_range=(40e-9,120e-9), subplots=True)
plotter.plot_signals("cs_signals_code_5bit.csv",filters=["d0=0,d1=0,d2=0,d3=0,d4=0", "d0=0,d1=0,d2=0,d3=1,d4=1", "d0=0,d1=0,d2=0,d3=0,d4=1", "d0=1,d1=1,d2=1,d3=0,d4=0", "d0=1,d1=1,d2=1,d3=1,d4=0"], t_range=(40e-9,120e-9), subplots=True )

# 3. plot lineartity
plotter.plot_linearity("cs_delay_code_4bit.csv")
plotter.plot_linearity("cs_delay_code_5bit.csv")
plotter.plot_linearity("cs_delay_code_5bit_coarse.csv")
plotter.plot_linearity("cs_delay_code_5bit_nonidealcurs.csv")

# 4. PVT analysis 
plotter.plot_pvt_sweep("cs_delay_power_corner_T.csv", subfigure=False)
plotter.plot_pvt_sweep("cs_delay_power_VDD_corner_T.csv", VDD=True, subfigure=True)
plotter.plot_pvt_linearity("cs_delay_power_corner_T.csv", subplots= False)


#5. plot histogram for resolution and dynamic range
plotter.plot_histogram(["cs_delay_mc_tt_00000.csv", "cs_delay_mc_tt_11111.csv"])

# Same for variable slope
# 1. Automatic Plots (Only for sweeps)
files = [
    "dlcsi_delay_code_5bit.csv", "dlcsi_power_code_5bit.csv"]
for f in files:
    plotter.smart_plot(f)

# 2. plot transients


# 3. plot lineartity
plotter.plot_linearity("dlcsi_delay_code_5bit.csv")

# Same for variable slope
# 1. Automatic Plots (Only for sweeps)
files = [
    "pi_delay_code_4bit.csv", "pi_power_code_4bit.csv"]
for f in files:
    plotter.smart_plot(f)

# 2. plot transients


# 3. plot lineartity
plotter.plot_linearity("pi_delay_code_4bit.csv")