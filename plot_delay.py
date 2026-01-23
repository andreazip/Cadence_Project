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
    def __init__(self, base_dir="results_cadence", plot_dir="plots"):
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
        fname_lower = str(filename).lower()
        
        if "vs" in fname_lower:
            subfolder = "variable_slope"
        elif "cs" in fname_lower:
            subfolder = "constant_slope"
        
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

    def _reconstruct_digital_data(self, df, signal_name, bit_count):
        """Generic reconstruction for any bit depth."""
        data = []
        # Find all columns for this signal
        x_cols = [c for c in df.columns if c.startswith(signal_name) and c.endswith(' X')]
        if not x_cols:
            return None

        for x_col in x_cols:
            y_col = x_col[:-2] + ' Y'
            # Extract bits from header: d0=0, d1=1...
            header_bits = re.findall(r"[db](\d+)=(\d+)", x_col)
            header_val = 0
            found_indices = []
            for b_idx, b_val in header_bits:
                idx, val = int(b_idx), int(b_val)
                header_val |= (val << idx)
                found_indices.append(idx)
            
            # Identify the swept bit index (the one missing from the header)
            all_indices = set(range(bit_count))
            missing = sorted(list(all_indices - set(found_indices)))
            sweep_bit_idx = missing[0] if missing else bit_count - 1

            temp = df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
            for _, row in temp.iterrows():
                x_val = int(row[x_col])
                
                # Special Inversion Logic for 5-bit (as per user request)
                # If 5th bit is being swept, d4 = NOT(x)
                if bit_count == 5 and sweep_bit_idx == 4:
                    d_sweep = 1 - x_val
                else:
                    d_sweep = x_val

                code = header_val | (d_sweep << sweep_bit_idx)
                data.append({'code': code, 'y': row[y_col]})
        
        if not data: return None
        plot_df = pd.DataFrame(data).sort_values('code').drop_duplicates('code').reset_index(drop=True)
        
        # --- NEW LOGIC: Remove Code 16 if redundant with Code 15 ---
        if 16 in plot_df['code'].values and 0 in plot_df['code'].values:
            y0 = plot_df.loc[plot_df['code'] == 15, 'y'].values[0]
            y16 = plot_df.loc[plot_df['code'] == 16, 'y'].values[0]
            if np.isclose(y0, y16, rtol=1e-3):
                plot_df = plot_df[plot_df['code'] != 16].reset_index(drop=True)
        plot_df['label'] = plot_df['code'].apply(lambda c: bin(c)[2:].zfill(bit_count))
        return plot_df

    def plot_digital_sweep(self, filename, signal_name=None):
        df, path = self.load_data(filename)
        if df is None: return
        
        bit_count = self._get_bit_info(filename, df)
        if signal_name is None:
            signal_name = df.columns[0].split(' (')[0].split(' ')[0]

        plot_df = self._reconstruct_digital_data(df, signal_name, bit_count)
        if plot_df is None: return

        # Metrics
        y = plot_df['y'].values
        y_norm = y - y[0]
        dr = y_norm.max()
        res = (y[-1] - y[0]) / (len(y) - 1)

        plt.figure()
        unit = "W" if "P_" in signal_name or "power" in signal_name.lower() else "s"
        scale = 1e6 if unit == "W" else 1e9
        prefix = "u" if unit == "W" else "n"

        plt.plot(plot_df['label'], y_norm * scale, marker='o', color='crimson', 
                 label=f"DR: {dr*scale:.2f}{prefix}{unit}\nRes: {res*scale*1000:.1f}p{unit}")
        
        plt.title(f"{bit_count}-Bit Sweep: {signal_name}")
        plt.xlabel(f"Digital Code (LSB={bit_count-1}...0)")
        plt.ylabel(f"Relative Value ({prefix}{unit})")
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

        plot_df = self._reconstruct_digital_data(df, signal_name, bit_count)
        if plot_df is None: return

        y = plot_df['y'].values
        lsb_ideal = (y[-1] - y[0]) / (len(y) - 1)
        dnl = np.insert(np.diff(y) / lsb_ideal - 1, 0, 0)
        inl = np.cumsum(dnl)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        ax1.bar(plot_df['label'], dnl, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_ylabel("DNL (LSB)")
        ax1.set_title(f"{bit_count}-Bit Linearity: {signal_name} (LSB={lsb_ideal:.2e})")
        
        ax2.plot(plot_df['label'], inl, marker='o', color='crimson', markersize=4)
        ax2.set_ylabel("INL (LSB)")
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()

        save_path = self._get_save_path(filename, signal_name, "linearity")
        plt.savefig(save_path)

        plt.close()

    def plot_histogram(self, filename):
        """Plots a histogram for Monte Carlo data with Mean and Std Dev."""
        df, path = self.load_data(filename)
        if df is None: return None
        data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        plt.figure()
        plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        mean, std = data.mean(), data.std()
        plt.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean:.3e}')
        plt.axvline(mean + std, color='orange', linestyle='--', linewidth=1.5, label=f'Std Dev: {std:.3e}')
        plt.axvline(mean - std, color='orange', linestyle='--')
        plt.axvspan(mean - std, mean + std, color='orange', alpha=0.1, label='1-$\sigma$ Spread')
        plt.title(f"Monte Carlo: {path.name}\n($\mu$={mean:.3e}, $\sigma$={std:.3e})")
        plt.xlabel(df.columns[0].replace(' X', '')); plt.ylabel("Frequency")
        plt.legend(); plt.grid(True); plt.tight_layout()

        save_path = self._get_save_path(filename, path.name ,"mc")
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
    "cs_delay_code_4bit.csv", "cs_delay_code_5bit.csv", "cs_power_code_4bit.csv", "cs_power_code_5bit.csv"]
for f in files:
    plotter.smart_plot(f)

# 2. plot transients
plotter.plot_signals("cs_Pinst_4bit.csv")
plotter.plot_signals("cs_signals_code_4bit.csv")
plotter.plot_signals("cs_Pinst_5bit.csv")
plotter.plot_signals("cs_signals_code_5bit.csv",filters=["d0=0,d1=0,d2=0,d3=0,d4=0", "d0=0,d1=0,d2=0,d3=1,d4=1", "d0=0,d1=0,d2=0,d3=0,d4=1", "d0=1,d1=1,d2=1,d3=0,d4=0", "d0=1,d1=1,d2=1,d3=1,d4=0"], t_range=(40e-9,120e-9), subplots=True )

# 3. plot lineartity
plotter.plot_linearity("cs_delay_code_4bit.csv")
plotter.plot_linearity("cs_delay_code_5bit.csv")

# Same for variable slope
# 1. Automatic Plots (Only for sweeps)
files = [
    "vs_delay_code_3bit.csv", "vs_delay_invstrength.csv", "vs_delay_Cap.csv", "vs_delay_vdd.csv", "vs_delay_mc_tt.csv",
    "vs_power_code_3bit.csv", "vs_power_invstrength.csv", "vs_power_Cap.csv", "vs_power_vdd.csv", "vs_power_mc_tt.csv",
    "vs_delay_T_corner.csv", "vs_power_T_corner.csv"]
for f in files:
    plotter.smart_plot(f)

# 2. plot transients
plotter.plot_signals("vs_signals_Cap.csv", t_range=[40e-9, 80e-9], subplots=True)
plotter.plot_signals("vs_signals_code_3bit.csv", t_range=[40e-9, 80e-9], subplots=True)

# 3. plot lineartity
plotter.plot_linearity("vs_delay_code_3bit.csv")

