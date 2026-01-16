import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

# --- User Specified Style ---
PPT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (12, 7),
    "lines.linewidth": 2.5,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
}
plt.rcParams.update(PPT_STYLE)

class CadencePlotter:
    def __init__(self, base_dir="results_cadence"):
        self.base_dir = Path(base_dir)

    def _resolve_path(self, filename):
        path = Path(filename)
        if not path.exists():
            path = self.base_dir / filename
        return path

    def load_data(self, filename):
        path = self._resolve_path(filename)
        if not path.exists():
            return None, path
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df, path

    def plot_signals(self, filename):
        """Format: Multiple X-Y pairs (Transient waveforms)."""
        df, path = self.load_data(filename)
        if df is None: return None
        
        x_cols = [c for c in df.columns if c.endswith(' X')]
        plt.figure()
        for x_col in x_cols:
            y_col = x_col[:-2] + ' Y'
            if y_col in df.columns:
                x_vals = pd.to_numeric(df[x_col], errors='coerce')
                y_vals = pd.to_numeric(df[y_col], errors='coerce')
                mask = x_vals.notna() & y_vals.notna()
                plt.plot(x_vals[mask], y_vals[mask], label=x_col[:-2].lstrip('/'))
        
        plt.xlabel("X-axis"); plt.ylabel("Y-axis")
        plt.title(f"Waveforms: {path.name}")
        if len(x_cols) > 1: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"plots/signals_{path.stem}.png")

    def plot_generic_sweep(self, filename):
        """Format: Single X and Y pair."""
        df, path = self.load_data(filename)
        if df is None: return None
        x_cols = [c for c in df.columns if c.endswith(' X')]
        y_cols = [c for c in df.columns if c.endswith(' Y')]
        if not x_cols or not y_cols: return None
        
        plt.figure()
        plt.plot(df[x_cols[0]], df[y_cols[0]], marker='o')
        plt.xlabel(x_cols[0].replace(' X', '')); plt.ylabel(y_cols[0].replace(' Y', ''))
        plt.title(f"Sweep: {path.stem}")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"plots/generic_{path.stem}.png")

    def plot_digital_sweep(self, filename, y_label="Value"):
        """Format: Reconstructs b2b1b0 digital code."""
        df, path = self.load_data(filename)
        if df is None: return None
        data = []
        bit_pattern = re.compile(r"b0[^\d]*(\d)[^\d]*b1[^\d]*(\d)")
        for x_col in [c for c in df.columns if c.endswith(' X')]:
            y_col = x_col[:-2] + ' Y'
            m = bit_pattern.search(x_col)
            if m and y_col in df.columns:
                b0, b1 = int(m.group(1)), int(m.group(2))
                for _, row in df.iterrows():
                    try:
                        b2 = int(float(row[x_col]))
                        data.append({'c': (b2<<2)|(b1<<1)|b0, 'l': f"{b2}{b1}{b0}", 'y': row[y_col]})
                    except: continue
        if not data: return None
        plot_df = pd.DataFrame(data).sort_values('c')
        plt.figure()
        plt.plot(plot_df['l'], plot_df['y'], marker='o', color='crimson')
        plt.xlabel("Digital Code ($b_2 b_1 b_0$)"); plt.ylabel(y_label)
        plt.title(f"Digital Sweep: {path.name}")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"plots/digital_{path.stem}.png")

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
        plt.savefig(f"plots/mc_hist_{path.stem}.png")

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
        plt.savefig(f"plots/corner_{path.stem}.png")

    def smart_plot(self, filename):
        """Automatically detects format and plots accordingly."""
        df, path = self.load_data(filename)
        if df is None: return
        name_lower = filename.lower()
        if "mc" in name_lower: return self.plot_histogram(filename)
        if "corner_t" in name_lower: return self.plot_corner_temperature_sweep(filename)
        if any(re.search(r"b0.*b1", c) for c in df.columns): return self.plot_digital_sweep(filename)
        x_cols = [c for c in df.columns if c.endswith(' X')]
        if len(x_cols) == 1: return self.plot_generic_sweep(filename)
        if len(x_cols) > 1: return self.plot_signals(filename)

# --- Usage ---
plotter = CadencePlotter(base_dir="results_cadence")

# 1. Automatic Plots
files = [
    "delay_vs_code.csv", "power_vs_code.csv", "delay_vs_vdd.csv", 
    "delay_vs_C.csv", "power_vs_C.csv", "power_vs_vdd.csv", "plot_signals.csv",
    "delay_vs_corner_T.csv", "Power_vs_corner_T.csv", "power_mc_tt.csv", "delay_mc_tt.csv"
]
for f in files:
    plotter.smart_plot(f)

# 2. Specific Signal Plots
plotter.plot_signals("plot_signals_code.csv")