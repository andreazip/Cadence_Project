import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import SCIENCE_STYLE_OVERRIDES, apply_science_style, _multi_panel_figsize, maybe_suptitle, maybe_title, _multi_panel_figsize


apply_science_style()


SAVE_DIR = Path(
    r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\python simulation\coarse_fine_processed"
)


def codes_without_middle(n_codes: int) -> np.ndarray:
    """Match coarse_fine_dtc._codes_without_middle convention."""
    codes = np.arange(n_codes - 1, dtype=int)
    mid = (len(codes) - 1) // 2
    return np.delete(codes, mid)

def codes_with_middle(n_codes: int) -> np.ndarray:
    """Match coarse_fine_dtc._codes_without_middle convention."""
    return  np.arange(n_codes, dtype=int)

def split_into_coarse_blocks(y: np.ndarray, coarse_codes: int, fine_codes: int) -> list[np.ndarray]:
    """
    Split combined y data into coarse blocks.

    Expected lengths:
    - coarse_codes * fine_codes
    - coarse_codes * fine_codes - 2 (first and last block have one fewer sample)
    """
    total_expected = coarse_codes * fine_codes
    n = len(y)

    if n == total_expected:
        sizes = [fine_codes] * coarse_codes
    elif n == total_expected - 2:
        sizes = [fine_codes - 1] + [fine_codes] * (coarse_codes - 2) + [fine_codes - 1]
    else:
        raise ValueError(
            f"Cannot map {n} points to coarse/fine grid ({coarse_codes}x{fine_codes}). "
            f"Expected {total_expected} or {total_expected - 2}."
        )

    blocks = []
    start = 0
    for sz in sizes:
        stop = start + sz
        blocks.append(y[start:stop])
        start = stop

    return blocks


# def estimate_coarse_period(blocks: list[np.ndarray], coarse_active: np.ndarray, fine_idx_last: np.ndarray) -> float:
#     """Estimate coarse period from adjacent coarse-code delays."""
#     coarse_delays = []
#     for c in coarse_active:
#         b = blocks[int(c)]
#         idx = fine_idx_last[fine_idx_last < len(b)]
#         if len(idx) == 0:
#             coarse_delays.append(float(np.min(b)))
#         else:
#             coarse_delays.append(float(np.min(b[idx])))

#     d = np.abs(np.diff(np.asarray(coarse_delays, dtype=float)))
#     d = d[d > 0.0]
#     if len(d) == 0:
#         raise ValueError("Unable to estimate coarse period from data")
#     return float(np.median(d))


# def build_fine_profile(blocks: list[np.ndarray], fine_codes: int) -> np.ndarray:
#     """Build a global fine-delay profile by median across coarse blocks per fine index."""
#     profile = np.full(fine_codes, np.nan, dtype=float)
#     for i in range(fine_codes):
#         vals = [float(b[i]) for b in blocks if i < len(b)]
#         if len(vals) > 0:
#             profile[i] = float(np.median(vals))
#     return profile


def block_start_indices(blocks: list[np.ndarray]) -> np.ndarray:
    """Return global start index of each coarse block in the flattened input sequence."""
    starts = np.zeros(len(blocks), dtype=int)
    running = 0
    for i, b in enumerate(blocks):
        starts[i] = running
        running += len(b)
    return starts


# def fine_boundary_policy(remove, blocks: list[np.ndarray], coarse_active: np.ndarray, fine_codes: int) -> tuple[np.ndarray, dict]:
#     """Mirror CoarseFineDTC._fine_boundary_policy on measured data."""
#     fine_idx_last = codes_without_middle(fine_codes) if remove == True else codes_with_middle(fine_codes)
#     #coarse_period_s = estimate_coarse_period(blocks, coarse_active, fine_idx_last)
#     #fine_profile = build_fine_profile(blocks, fine_codes)
#     fine_codes_active = fine_idx_last[np.isfinite(fine_profile[fine_idx_last])]
#     fine_delays_active = fine_profile[fine_codes_active]
#     fine_min = float(np.min(fine_delays_active))
#     fine_span_active = fine_delays_active - fine_min

#     # _valid_fine_indices equivalent
#     valid_local = np.where((fine_span_active >= 0.0) & (fine_span_active <= coarse_period_s))[0]
#     fine_idx_regular = fine_codes_active[valid_local]

#     fine_codes_set = set(int(c) for c in fine_idx_regular)
#     valid_mask = np.array([int(c) in fine_codes_set for c in fine_codes_active], dtype=bool)
#     valid_spans = fine_span_active[valid_mask]
#     if len(valid_spans) == 0:
#         return fine_idx_regular, {
#             "coarse_period_s": coarse_period_s,
#             "resolution_s": 0.0,
#             "residual_s": 0.0,
#         }

#     fine_steps = np.diff(fine_span_active)
#     fine_steps = fine_steps[fine_steps > 0.0]
#     res_s = float(np.median(fine_steps)) if len(fine_steps) > 0 else 0.0
#     residual_s = float(coarse_period_s - np.max(valid_spans))

#     meta = {
#         "coarse_period_s": coarse_period_s,
#         "resolution_s": res_s,
#         "residual_s": residual_s,
#     }
#     return fine_idx_regular, meta


def combine_like_coarse_fine_dtc(
    blocks: list[np.ndarray],
    coarse_codes: int,
    fine_codes: int,
    max_boundary_skip: int = -1,
    remove_coarse: bool = False,
    remove_fine: bool = False,  
    slope_negative: bool = False
):
    """
    New requested policy:
    - First coarse segment: keep all available fine points.
        - Each next coarse segment: start from the first point < previous - 0.5*LSB_local.
        - Each next point must be >= previous - 0.45*LSB_local, otherwise skip more until this is satisfied.
      LSB_local is the average fine step magnitude in that coarse segment.
    """
    coarse_active = codes_without_middle(coarse_codes) if remove_coarse == True else codes_with_middle(coarse_codes)
    fine_idx_last = codes_without_middle(fine_codes) if remove_fine == True else codes_with_middle(fine_codes)

    # _, meta = fine_boundary_policy(remove, blocks, coarse_active, fine_codes)

    combined = []
    coarse_marker = []
    coarse_codes_out = []
    fine_codes_out = []
    selected_input_indices = []
    candidate_local = np.array([], dtype=int)
    boundary_skip_count = 0
    boundary_no_solution_count = 0
    boundary_violation_count = 0
    boundary_lsb_values = []
    boundary_margin_values = []
    boundary_details = []
    previous_values = []
    starts = block_start_indices(blocks)
    # Determine slope to know if we are looking for a value higher or lower
    is_decreasing = slope_negative

   

    local_lsb = []
    
    for c in coarse_active:
        block = blocks[int(c)] 
        fine_idx = fine_idx_last[fine_idx_last < len(block)]

        # 1) USE RAW DATA DIRECTLY
        # Instead of building local_values from a profile, use the actual block data
        local_values = block[fine_idx].astype(float)
        if len(local_values) > 1:
            local_lsb.append((local_values[-1] - local_values[0])/(len(local_values)-1))
        else:
            local_lsb = 0.0
        
    lsb = np.abs(float(np.mean(np.array(local_lsb))))

    #configure the first fine block
    block = blocks[int(coarse_active[0])]
    previous_idx = fine_idx_last[fine_idx_last < len(block)]
    previous_values = block[previous_idx].astype(float)

    coarse_active = coarse_active[1:]  # Start from the second coarse code since the first is fully included

    for c_code in coarse_active:
        candidate_local = np.array([], dtype=int)
        iterate = True

        block_prev = block
        block = blocks[int(c_code)] 
        fine_idx = fine_idx_last[fine_idx_last < len(block)]

        # 1) USE RAW DATA DIRECTLY
        # Instead of building local_values from a profile, use the actual block data
        local_values = block[fine_idx].astype(float)

        # 2) CONNECT USING HALF-LSB ERROR
        if c_code > 0 and len(local_values) > 0 and len(fine_idx) > 0:
            for (i,p) in enumerate(previous_values):
                if iterate:
                    p = float(p)
                    if is_decreasing:
                        margin = p - local_values
                    else:
                        margin = local_values - p
                    
                    candidate = np.where((margin >= 0.6 * lsb) & (margin <= 1.4 * lsb))[0]

                    if len(candidate) > 0 and len(candidate_local) == 0 and candidate[0] > 3:
                            prev_value = p
                            end_previous = i
                            candidate_local = candidate
                            iterate = False

            if iterate:
                prev_value = previous_values[-1]
                end_previous = len(previous_values) - 1
                start_local = 0
                boundary_no_solution_count += 1
                print("I am here")
                if is_decreasing:
                    lsb = - lsb
                candidate_local = local_values - (prev_value + lsb)
                start_local = np.argmin(np.abs(candidate_local))
            else:    
                start_local = int(candidate_local[0]) 
            
           
            if max_boundary_skip >= 0:
                start_local = min(start_local, max_boundary_skip)

            #update the indices
            previous_idx = previous_idx[:end_previous+1] 
            previous_values = previous_values[:end_previous+1]
            fine_idx = fine_idx[start_local:]
            local_values = local_values[start_local:]
            boundary_skip_count += int(start_local)

            if len(local_values) == 0:
                violates_after = True
                margin = float("inf")
                next_first_value = float("nan")
            else:
                next_first_value = float(local_values[0])
                prev_value = float(previous_values[end_previous])
                margin = float(next_first_value - prev_value) 
                
                if slope_negative:
                    violates_after = bool((-margin < 0.5*lsb) or (-margin > 1.5*lsb))
                else:
                    violates_after = bool((margin < 0.5*lsb) or (margin > 1.5*lsb))
                
            if violates_after:
                boundary_violation_count += 1
                print(f"Boundary violation after policy for coarse code {c_code}: next_first_value={next_first_value:.6e} violates previous value={prev_value:.6e} (margin={margin:.6e}).")

            boundary_lsb_values.append(float(lsb))
            boundary_margin_values.append(float(margin))
            boundary_details.append(
                {
                    "coarse_from": int(c_code-1),
                    "coarse_to": int(c_code),
                    "local_lsb": float(lsb),
                    "previous_value": prev_value,
                    "next_value_used": float(next_first_value),
                    "margin_after": float(margin),
                    "skips": int(start_local),
                    "violates_after": bool(violates_after),
                }
            )

        coarse_base = float(np.min(block))
        for local_i, f_code in enumerate(previous_idx):

            value = float(block_prev[int(f_code)]) # Raw measurement
            combined.append(value)
            coarse_marker.append(local_i == 0)
            coarse_codes_out.append(int(c_code-1))
            fine_codes_out.append(int(f_code))
            selected_input_indices.append(int(starts[int(c_code-1)] + int(f_code)))
        
        previous_idx = fine_idx
        previous_values = local_values
    
    block = blocks[int(coarse_active[-1])]
    
    for local_i, f_code in enumerate(previous_idx):
        value = float(block[int(f_code)]) # Raw measurement
        combined.append(value)
        coarse_marker.append(local_i == 0)
        coarse_codes_out.append(int(c_code))
        fine_codes_out.append(int(f_code))
        selected_input_indices.append(int(starts[int(coarse_active[-1])] + int(f_code)))


    meta = {
        "coarse_active_len": int(len(coarse_active)),
        "fine_regular_len": int(len(fine_idx_last)),
        "fine_last_len": int(len(fine_idx_last)),
        "boundary_skip_count": int(boundary_skip_count),
        "boundary_lsb_mean": float(np.mean(boundary_lsb_values)) if len(boundary_lsb_values) > 0 else 0.0,
        "boundary_margin_max": float(np.max(boundary_margin_values)) if len(boundary_margin_values) > 0 else 0.0,
        "boundary_violation_count": int(boundary_violation_count),
        "boundary_no_solution_count": int(boundary_no_solution_count),
        "max_boundary_skip": int(max_boundary_skip),
        "boundary_details": boundary_details,
        "remove_coarse": bool(remove_coarse),
        "remove_fine": bool(remove_fine),   
    }

    return (
        np.asarray(combined, dtype=float),
        np.asarray(coarse_marker, dtype=bool),
        np.asarray(coarse_codes_out),
        np.asarray(fine_codes_out),
        np.asarray(selected_input_indices, dtype=int),
        meta,
    )


def compute_dnl_inl(delay: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute DNL/INL in LSB from delay characteristic."""
    if len(delay) < 2:
        return np.array([]), np.array([]), 0.0

    lsb = float((delay[-1] - delay[0]) / (len(delay) - 1))

    if lsb == 0.0:
        return np.array([]), np.array([]), lsb

    dnl = np.insert(np.diff(delay) / lsb - 1.0, 0, 0.0)
    inl = np.cumsum(dnl)
    return dnl, inl, lsb


def _load_power_trace_from_csv(power_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one transient power CSV and return finite (time_s, |power_w|)."""
    df = pd.read_csv(power_csv)
    df.columns = [c.strip() for c in df.columns]

    time = pd.to_numeric(df[df.columns[0]], errors="coerce").to_numpy(dtype=float)
    power = pd.to_numeric(df[df.columns[1]], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(power)
    time = time[valid]
    power = np.abs(power[valid])

    if len(time) < 2:
        raise ValueError(f"Not enough time samples in {power_csv}")

    return time, power


def _load_power_trace_sequence(power_csv_list: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate multiple power traces in provided order.

    Each file is re-based to start at t=0, then shifted to start after the previous
    file ends, preserving local sampling cadence.
    """
    all_time = []
    all_power = []
    t_offset = 0.0

    for i, p in enumerate(power_csv_list):
        time, power = _load_power_trace_from_csv(p)
        dt_med = float(np.median(np.diff(time))) if len(time) > 1 else 0.0

        # Rebase each trace and append after the previous one.
        time_rebased = time - float(time[0])
        if i > 0:
            time_rebased = time_rebased + t_offset

        all_time.append(time_rebased)
        all_power.append(power)

        t_offset = float(time_rebased[-1] + max(dt_med, 0.0))

    return np.concatenate(all_time), np.concatenate(all_power)


def compute_average_power_per_period(
    power_csv_list: list[Path],
    n_periods: int,
    period_s: float,
    start_time_s: float | None = None,
) -> np.ndarray:
    """Compute average power for each execution period using trapezoidal integration."""
    time, power = _load_power_trace_sequence(power_csv_list)

    if start_time_s is None:
        start_time_s = float(time[0])

    avg_power = np.full(n_periods, np.nan, dtype=float)
    for i in range(n_periods):
        t0 = start_time_s + i * period_s
        t1 = t0 + period_s
        m = (time >= t0) & (time < t1)
        if np.count_nonzero(m) < 2:
            continue
        tw = time[m]
        pw = power[m]
        duration = float(tw[-1] - tw[0])
        if duration <= 0.0:
            continue
        energy = float(np.trapz(pw, tw))
        avg_power[i] = energy / duration

    return avg_power


def plot_selected_average_power(avg_power_w: np.ndarray, out_dir: Path, P_static: float) -> Path:
    """Plot average power across selected linearized coarse-fine codes."""
    if len(avg_power_w) > 1:
        # Remove the first plotted point to suppress the initial jump artifact.
        avg_power_plot = avg_power_w[1:] - P_static
    else:
        avg_power_plot = avg_power_w - P_static

    codes = np.arange(len(avg_power_plot))
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(codes, avg_power_plot * 1e6, linewidth=1.8, color="#1F77B4")
    maybe_title(ax, "Average Power per 20 ns Execution (Linearized Coarse-Fine Codes)")
    ax.set_xlabel(r"$\mathrm{Code}$")
    ax.set_ylabel(r"$P_{\mathrm{tot}} [\mu W]$")
    ax.grid(True, alpha=0.35)

    out = out_dir / "processed_avg_power_selected_like_coarse_fine_dtc.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def filter_power_codes_for_plot(
    coarse_codes: np.ndarray,
    fine_codes: np.ndarray,
    source_period_index: np.ndarray,
    avg_power_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only unique fine codes per coarse segment and remove fine code 0.

    This is applied to the power plot/export path only.
    """
    keep = np.zeros(len(avg_power_w), dtype=bool)
    seen_per_coarse: dict[int, set[int]] = {}

    for i, (c, f) in enumerate(zip(coarse_codes, fine_codes)):
        c_i = int(c)
        f_i = int(f)
        if f_i == 0:
            continue
        if c_i not in seen_per_coarse:
            seen_per_coarse[c_i] = set()
        if f_i in seen_per_coarse[c_i]:
            continue
        seen_per_coarse[c_i].add(f_i)
        keep[i] = True

    return (
        coarse_codes[keep],
        fine_codes[keep],
        source_period_index[keep],
        avg_power_w[keep],
    )

def plot_coarse_fine_synthesis(self, coarse_file, fine_file, num_total_codes=8192, **kwargs):
        """
        Combines independent coarse delay steps and fine delay sweeps into a single 
        unified characteristic DataFrame across exactly 8192 codes.
        Supports both single-trace files and multi-corner PVT files by matching header tokens.
        Saves the synthesized dataset to CSV and exports a vector PDF plot.
        """
        df_c, _ = self.load_data(coarse_file)
        df_f, _ = self.load_data(fine_file)
        
        if df_c is None or df_f is None:
            print("Error: Missing one or both input dataframes.")
            return

        # Identify all column pairs for both files
        def get_header_pairs(df):
            pairs = {}
            cols = df.columns
            for i in range(0, len(cols), 2):
                if i + 1 < len(cols):
                    # Clean up the name token to use as a tracking alignment key
                    clean_name = cols[i+1].replace('delay (', '').replace(') Y', '').replace('t_delay ', '').strip()
                    pairs[clean_name] = (cols[i], cols[i+1])
            return pairs

        pairs_c = get_header_pairs(df_c)
        pairs_f = get_header_pairs(df_f)

        # Handle header matching alignment gracefully
        matched_keys = []
        if len(pairs_c) == 1 and len(pairs_f) == 1:
            # Single-trace mode fallback
            matched_keys = [(list(pairs_c.keys())[0], list(pairs_f.keys())[0])]
        else:
            # Multi-corner exact match mode
            for k in pairs_c:
                if k in pairs_f:
                    matched_keys.append((k, k))

        if not matched_keys:
            print("Error: Could not automatically align corner headers between coarse and fine files.")
            return

        output_data = {}
        codes = np.arange(num_total_codes)

        # Create figure using your standard class layout tools
        fig, ax = self._create_figure() if hasattr(self, '_create_figure') else plt.subplots(figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(matched_keys))))

        for idx, (k_c, k_f) in enumerate(matched_keys):
            x_col_c, y_col_c = pairs_c[k_c]
            x_col_f, y_col_f = pairs_f[k_f]
            
            c_y = pd.to_numeric(df_c[y_col_c], errors='coerce').dropna().values
            f_y = pd.to_numeric(df_f[y_col_f], errors='coerce').dropna().values
            
            if len(c_y) == 0 or len(f_y) == 0:
                continue
                
            # Compute the required fine trace size per coarse step to hit num_total_codes perfectly
            target_fine_len = num_total_codes // len(c_y)
            
            # Resample the fine delay vector to match the target segment length perfectly
            xp = np.linspace(0, 1, len(f_y))
            x_new = np.linspace(0, 1, target_fine_len)
            f_y_resampled = np.interp(x_new, xp, f_y)
            
            # Create composite delay array
            total_delay = []
            for c_val in c_y:
                for f_val in f_y_resampled:
                    total_delay.append(c_val + f_val)
            
            # Truncate or pad to fit exactly num_total_codes if any rounding remains
            total_delay = np.array(total_delay)[:num_total_codes]
            if len(total_delay) < num_total_codes:
                total_delay = np.pad(total_delay, (0, num_total_codes - len(total_delay)), 'edge')

            # Structure column layout to match standard Cadence trace pairs
            out_base = y_col_c.replace('t_delay Y', 'delay_composite')
            output_data[out_base + ' X'] = codes
            output_data[out_base + ' Y'] = total_delay
            
            # Scale output to nanoseconds (ns) for graph rendering readability
            label_text = k_c if len(matched_keys) > 1 else 'Composite Delay'
            ax.plot(codes, total_delay * 1e9, color=colors[idx], linewidth=2.2, label=label_text)

        # Export raw generated table directly to CSV
        synthesized_df = pd.DataFrame(output_data)
        out_csv_path = self.plot_dir / "synthesized_8192_delay.csv"
        synthesized_df.to_csv(out_csv_path, index=False)
        
        # Apply standard labels and grids
        if hasattr(self, '_format_plot_labels'):
            self._format_plot_labels(
                ax, 
                xlabel="Digital Input Code (0 to 8191)", 
                ylabel="Total Delay (ns)", 
                title="Synthesized 8192-Code Composite Characteristic"
            )
        else:
            ax.set_xlabel("Digital Input Code (0 to 8191)", fontweight='bold')
            ax.set_ylabel("Total Delay (ns)", fontweight='bold')
            ax.set_title("Synthesized 8192-Code Composite Characteristic", fontweight='bold')
            
        if hasattr(self, '_apply_grid_styling'):
            self._apply_grid_styling(ax, alpha=0.35)
        else:
            ax.grid(True, linestyle='--', alpha=0.4)
            
        if len(matched_keys) > 1:
            ax.legend(loc='best')

        plt.tight_layout()
        out_pdf_path = self.plot_dir / "synthesized_8192_delay_characteristic.pdf"
        plt.savefig(out_pdf_path, dpi=300)
        
        print(f"\n" + "="*70)
        print(f" SUCCESS: Generated full composite characteristics!")
        print(f" Saved CSV Data Asset: {out_csv_path}")
        print(f" Saved Vector PDF Plot: {out_pdf_path}")
        print(f" Total Code Count:      {len(synthesized_df)}")
        print("="*70 + "\n")


def plot_results(delay: np.ndarray, coarse_marker: np.ndarray, dnl: np.ndarray, inl: np.ndarray, out_dir: Path):
    """Save delay characteristic and DNL/INL plots."""
    x = np.arange(len(delay))

    fig1, ax1 = plt.subplots(constrained_layout=True)
    ax1.plot(x, delay*1e9, linewidth=2, label=r"$t_{\mathrm{delay}}$")
    maybe_suptitle(ax1, "Delay Characteristic (Processed Like coarse_fine_dtc)")
    
    ax1.set_ylabel(r"$t_{\mathrm{delay}}~[\mathrm{ns}]$", fontsize=15, fontweight='bold')
    ax1.set_xlabel(r" $\mathrm{Code}$", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    p1 = out_dir / "processed_delay_characteristic_like_coarse_fine_dtc.pdf"
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(x, dnl, linewidth=1.4)
    ax2.axhline(0.5, linestyle="--", linewidth=1.0)
    ax2.axhline(-0.5, linestyle="--", linewidth=1.0)
    maybe_suptitle(ax2, "DNL (Processed Like coarse_fine_dtc)")
    ax2.set_ylabel(r"$\mathrm{DNL} [\mathrm{LSB}]$", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig3, ax3 = plt.subplots()
    ax3.plot(x, inl, linewidth=1.4)
    ax3.axhline(0.0, linestyle="--", linewidth=1.0)
    maybe_suptitle(ax3, "INL (Processed Like coarse_fine_dtc)")
    ax3.set_xlabel(r"$\mathrm{Combined}$ $\mathrm{Code}$", fontsize=12, fontweight='bold')
    ax3.set_ylabel(r"$\mathrm{INL} [\mathrm{LSB}]$", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    p2 = out_dir / "processed_dnl_like_coarse_fine_dtc.pdf"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)

    p3 = out_dir / "processedinl_like_coarse_fine_dtc.pdf"
    fig3.savefig(p3, dpi=300)
    plt.close(fig3)

    return p1, p2, p3

    
def build_cli_parser():
    parser = argparse.ArgumentParser(description="Process CSV exactly like coarse_fine_dtc indexing/policy.")
    parser.add_argument("--csv", default="results_cadence/delay_coarse_fine.csv")
    parser.add_argument(
        "--power-csvs",
        nargs="+",
        default=[],
        help="Ordered power transient CSVs; traces are processed sequentially in this order.",
    )
    parser.add_argument("--out-dir", default=str(SAVE_DIR))
    parser.add_argument("--coarse-codes", type=int, default=256)
    parser.add_argument("--fine-codes", type=int, default=32)
    parser.add_argument("--period-s", type=float, default=20e-9)
    parser.add_argument("--power-start-time", type=float, default=None)
    parser.add_argument(
        "--max-boundary-skip",
        type=int,
        default=-1,
        help="Maximum skipped fine codes at each boundary (-1 means unlimited).",
    )
    parser.add_argument("--remove-coarse", type=bool, default=False)
    parser.add_argument("--remove-fine", type=bool, default=False)
    parser.add_argument(
        "--slope-negative", type=bool, default=False,
        help="Set to True if the delay characteristic is expected to decrease with increasing code."
    )
    parser.add_argument(
        "--static-power-uw", type=float, default=0.0,
        help="Estimated static power in microwatts to subtract from the average power plot."
    )
    parser.add_argument('--coarse_file', type=str, help='Path to the coarse delay CSV file')
    parser.add_argument('--fine_file', type=str, help='Path to the fine delay CSV file')
    parser.add_argument('--coarse_power_file', type=str, help='Path to coarse average power lookup file')
    parser.add_argument('--fine_power_file', type=str, help='Path to fine average power lookup file')

    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Detect Dataset Modes (MC vs Corner vs Standard) ---
    is_mc_run = False
    is_corner_run = False
    active_sets = []
    df_coarse, df_fine = None, None

    if args.coarse_file and args.fine_file:
        df_coarse = pd.read_csv(args.coarse_file)
        df_fine = pd.read_csv(args.fine_file)

        # Scan for Monte Carlo indicators
        def scan_mc(df):
            return [re.search(r'mcparamset=(\d+)', c).group(1) for c in df.columns if re.search(r'mcparamset=(\d+)', c)]
        mc_sets = sorted(list(set(scan_mc(df_coarse)) & set(scan_mc(df_fine))), key=int)

        # Scan for PVT Corner indicators (Cadence modelFiles syntax or explicit names)
        def scan_corners(df):
            corners = []
            for col in df.columns:
                if " X" in col or " Y" in col:
                    c_name = col.replace("delay (", "").replace(") X", "").replace(") Y", "").strip()
                    if c_name not in corners:
                        corners.append(c_name)
            return corners
        corner_sets = sorted(list(set(scan_corners(df_coarse)) & set(scan_corners(df_fine))))

        if len(mc_sets) > 1:
            is_mc_run = True
            active_sets = mc_sets
        elif len(corner_sets) > 1:
            is_corner_run = True
            active_sets = corner_sets

    # Global power data parsing trigger flag
    has_power_files = bool(args.coarse_power_file and args.fine_power_file)
    if has_power_files:
        df_p_coarse = pd.read_csv(args.coarse_power_file)
        df_p_fine = pd.read_csv(args.fine_power_file)

    # =========================================================================
    # MULTI-CORNER PVT EXECUTION BRANCH
    # =========================================================================
    if is_corner_run:
        print(f"\n[CORNER MODE] Processing multi-corner characteristics using policy analysis...")
        
        def parse_header(col_name):
            clean = col_name.replace('delay (modelFiles=toplevel.scs:', '').replace(') Y', '').strip()
            match = re.search(r'([^,]+),Vdd=([^,]+),temperature=([^)]+)', clean)
            if match:
                proc = match.group(1).replace('top_', '').upper()
                return proc, match.group(2), match.group(3)
            return None

        data_points = {}  # vdd -> condition_key -> (peak_dnl, peak_inl, dynamic_range, mean_power)
        found_procs = set()
        found_temps = set()
        
        columns = df_coarse.columns
        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                parsed = parse_header(columns[i+1])
                if not parsed:
                    continue
                proc, vdd, temp = parsed
                cond_key = f"{proc},{temp}"
                
                found_procs.add(proc)
                found_temps.add(temp)

                col_y_c = columns[i+1]
                col_y_f = col_y_c if col_y_c in df_fine.columns else df_fine.columns[i+1]

                c_y = pd.to_numeric(df_coarse[col_y_c], errors='coerce').dropna().values
                f_y = pd.to_numeric(df_fine[col_y_f], errors='coerce').dropna().values

                if len(c_y) == 0 or len(f_y) == 0:
                    continue

                num_total_codes = 8192
                target_fine_len = num_total_codes // len(c_y)
                xp = np.linspace(0, 1, len(f_y))
                x_new = np.linspace(0, 1, target_fine_len)
                f_y_resampled = np.interp(x_new, xp, f_y)

                total_delay = []
                for c_val in c_y:
                    for f_val in f_y_resampled:
                        total_delay.append(c_val + f_val)
                y_corner = np.array(total_delay)[:num_total_codes]

                blocks = split_into_coarse_blocks(y_corner, args.coarse_codes, args.fine_codes)
                delay, coarse_marker, coarse_codes_out, fine_codes_out, _, info = combine_like_coarse_fine_dtc(
                    blocks,
                    coarse_codes=args.coarse_codes,
                    fine_codes=args.fine_codes,
                    max_boundary_skip=args.max_boundary_skip,
                    remove_coarse=args.remove_coarse,
                    remove_fine=args.remove_fine,
                    slope_negative=args.slope_negative,
                )

                dnl, inl, _ = compute_dnl_inl(delay)
                
                if len(delay) >= 2:
                    peak_dnl = np.max(np.abs(dnl))
                    peak_inl = np.max(np.abs(inl))
                    dyn_range = (np.max(delay) - np.min(delay)) * 1e9

                    mean_p_uW = np.nan
                    if has_power_files:
                        p_c_vals = pd.to_numeric(df_p_coarse.iloc[:, 2], errors='coerce').dropna().values[:args.coarse_codes]
                        p_f_vals = pd.to_numeric(df_p_fine.iloc[:, 2], errors='coerce').dropna().values[:args.fine_codes]
                        target_fine_len_p = 8192 // len(p_c_vals)
                        xp_p = np.linspace(0, 1, len(p_f_vals))
                        x_new_p = np.linspace(0, 1, target_fine_len_p)
                        p_f_resampled = np.interp(x_new_p, xp_p, p_f_vals)

                        power_list = []
                        for c_hw, f_hw in zip(coarse_codes_out, fine_codes_out):
                            c_idx = int(c_hw) if int(c_hw) < len(p_c_vals) else -1
                            f_idx = int(f_hw) if int(f_hw) < len(p_f_resampled) else -1
                            power_list.append(p_c_vals[c_idx] + p_f_resampled[f_idx])
                        mean_p_uW = float(np.mean(power_list))
                    
                    if vdd not in data_points:
                        data_points[vdd] = {}
                    data_points[vdd][cond_key] = (peak_dnl, peak_inl, dyn_range, mean_p_uW)

                    lut_path = out_dir / f"dtc_calibrated_hardware_lut_corner_{proc}_{vdd}V_{temp}C.csv"
                    lut_payload = {
                        "linearized_code": np.arange(len(delay)),
                        "coarse_hardware_code": coarse_codes_out.astype(int),
                        "fine_hardware_code": fine_codes_out.astype(int),
                        "expected_delay_s": delay
                    }
                    if has_power_files:
                        lut_payload["expected_total_power_uW"] = np.array(power_list)
                    pd.DataFrame(lut_payload).to_csv(lut_path, index=False)

        sorted_procs = sorted(list(found_procs))
        def temp_sort_key(t_str):
            try: return float(t_str)
            except ValueError: return 999.0
        sorted_temps = sorted(list(found_temps), key=temp_sort_key)

        all_conditions = []
        for p in sorted_procs:
            for t in sorted_temps:
                c_str = f"{p},{t}"
                if any(c_str in data_points[v] for v in data_points):
                    all_conditions.append(c_str)

        x_indices = np.arange(len(all_conditions))
        vdd_colors = {'0.88': plt.cm.tab10(0), '1.1': plt.cm.tab10(1), '1.32': plt.cm.tab10(2)}

        # --- APPLY GLOBAL PACKAGED STYLING BOUNDS ---
        with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
            # --- Figure 1: Peak DNL Trend Plot ---
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for vdd, cond_dict in data_points.items():
                color = vdd_colors.get(vdd, 'black')
                y_pts = [cond_dict[c][0] if c in cond_dict else np.nan for c in all_conditions]
                ax1.plot(x_indices, y_pts, color=color, linestyle='-', marker='o', linewidth=2, label=rf'$V_{{\mathrm{{dd}}}}={vdd}$ V')
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(all_conditions, rotation=35, ha='right')
            ax1.set_xlabel("PVT Operating Corners")
            ax1.set_ylabel(r"Peak |$DNL$| (LSB)")
            ax1.set_title("Peak DNL Variation Across Calibrated Corners")
            ax1.grid(True, linestyle='--', alpha=0.35)
            ax1.legend(loc='best')
            plt.tight_layout()
            plt.savefig(out_dir / "peak_dnl_trends_comparison.pdf", dpi=300)
            plt.close()

            # --- Figure 2: Peak INL Trend Plot ---
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for vdd, cond_dict in data_points.items():
                color = vdd_colors.get(vdd, 'black')
                y_pts = [cond_dict[c][1] if c in cond_dict else np.nan for c in all_conditions]
                ax2.plot(x_indices, y_pts, color=color, linestyle='-', marker='s', linewidth=2, label=rf'$V_{{\mathrm{{dd}}}}={vdd}$ V')
            ax2.set_xticks(x_indices)
            ax2.set_xticklabels(all_conditions, rotation=35, ha='right')
            ax2.set_xlabel("PVT Operating Corners")
            ax2.set_ylabel(r"Peak |$INL$| (LSB)")
            ax2.set_title("Peak INL Variation Across Calibrated Corners")
            ax2.grid(True, linestyle='--', alpha=0.35)
            ax2.legend(loc='best')
            plt.tight_layout()
            plt.savefig(out_dir / "peak_inl_trends_comparison.pdf", dpi=300)
            plt.close()

            # --- Figure 3: Dynamic Tuning Range Scaling Variation Trend ---
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for vdd, cond_dict in data_points.items():
                color = vdd_colors.get(vdd, 'black')
                y_pts = [cond_dict[c][2] if c in cond_dict else np.nan for c in all_conditions]
                ax3.plot(x_indices, y_pts, color=color, linestyle='-', marker='^', linewidth=2, label=rf'$V_{{\mathrm{{dd}}}}={vdd}$ V')
            ax3.set_xticks(x_indices)
            ax3.set_xticklabels(all_conditions, rotation=35, ha='right')
            ax3.set_xlabel("PVT Operating Corners")
            ax3.set_ylabel(r"Dynamic Tuning Range $\Delta t_{\mathrm{{max}}}$ (ns)")
            ax3.set_title("DTC Dynamic Tuning Range Scaling Across Corners")
            ax3.grid(True, linestyle='--', alpha=0.35)
            ax3.legend(loc='best')
            plt.tight_layout()
            plt.savefig(out_dir / "dynamic_range_trends_comparison.pdf", dpi=300)
            plt.close()

            # --- Figure 4: Corner Power Metric Drift Profile ---
            if has_power_files:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                for vdd, cond_dict in data_points.items():
                    color = vdd_colors.get(vdd, 'black')
                    y_pts = [cond_dict[c][3] if c in cond_dict else np.nan for c in all_conditions]
                    ax4.plot(x_indices, y_pts, color=color, linestyle='-', marker='d', linewidth=2, label=rf'$V_{{\mathrm{{dd}}}}={vdd}$ V')
                ax4.set_xticks(x_indices)
                ax4.set_xticklabels(all_conditions, rotation=35, ha='right')
                ax4.set_xlabel("PVT Operating Corners")
                ax4.set_ylabel(r"Mean Calibrated Power $P_{\mathrm{{tot}}}$ ($\mu$W)")
                ax4.set_title("DTC Calibration Average Power Dissipation Drift Trends")
                ax4.grid(True, linestyle='--', alpha=0.35)
                ax4.legend(loc='best')
                plt.tight_layout()
                plt.savefig(out_dir / "corner_calibrated_power_trends.pdf", dpi=300)
                plt.close()

        print(f"\nSUCCESS: Processing Complete across Corners.")
        return

    # =========================================================================
    # MULTI-REALIZATION MONTE CARLO EXECUTION BRANCH
    # =========================================================================
    elif is_mc_run:
        print(f"\n[MONTE CARLO] Found {len(active_sets)} realizations. Loop-applying policy...")
        all_dnl_traces, all_inl_traces = [], []
        max_output_length = 0

        for ps in active_sets:
            match_cols_c = sorted([c for c in df_coarse.columns if f"mcparamset={ps}" in c])
            match_cols_f = sorted([c for c in df_fine.columns if f"mcparamset={ps}" in c])

            c_y = pd.to_numeric(df_coarse[match_cols_c[1]], errors='coerce').dropna().values
            f_y = pd.to_numeric(df_fine[match_cols_f[1]], errors='coerce').dropna().values

            num_total_codes = 8192
            target_fine_len = num_total_codes // len(c_y)
            xp = np.linspace(0, 1, len(f_y))
            x_new = np.linspace(0, 1, target_fine_len)
            f_y_resampled = np.interp(x_new, xp, f_y)

            total_delay = []
            for c_val in c_y:
                for f_val in f_y_resampled: total_delay.append(c_val + f_val)
            y_mc = np.array(total_delay)[:num_total_codes]

            blocks = split_into_coarse_blocks(y_mc, args.coarse_codes, args.fine_codes)
            delay, coarse_marker, coarse_codes_out, fine_codes_out, selected_input_indices, info = combine_like_coarse_fine_dtc(
                blocks, coarse_codes=args.coarse_codes, fine_codes=args.fine_codes,
                max_boundary_skip=args.max_boundary_skip, remove_coarse=args.remove_coarse,
                remove_fine=args.remove_fine, slope_negative=args.slope_negative,
            )

            dnl, inl, lsb = compute_dnl_inl(delay)
            all_dnl_traces.append(dnl)
            all_inl_traces.append(inl)
            max_output_length = max(max_output_length, len(delay))

        with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
            fig1,ax1 = plt.subplots(constrained_layout=True)

            for i, (dnl_t, inl_t) in enumerate(zip(all_dnl_traces, all_inl_traces)):
                codes_x = np.arange(len(dnl_t))
                ax1.plot(codes_x, dnl_t, color=plt.cm.tab10(0), alpha=0.2, linewidth=0.7)

            ax1.set_ylabel(r"$\mathrm{DNL}$ $[\mathrm{LSB}]$")
            ax1.hlines([0.5, -0.5], 0, max_output_length - 1, colors='gray', linestyles='--', linewidth=1)
            ax1.set_xlabel(r"$\mathrm{Combined}$ $\mathrm{Code}$")
            ax1.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "monte_carlo_dnl.pdf", dpi=300)
            plt.close(fig1)

            fig2,ax2 = plt.subplots(constrained_layout=True)
            for i, (dnl_t, inl_t) in enumerate(zip(all_dnl_traces, all_inl_traces)):
                codes_x = np.arange(len(dnl_t))
                ax2.plot(codes_x, inl_t, color=plt.cm.tab10(1), alpha=0.2, linewidth=0.7)

            ax2.set_ylabel(r"$\mathrm{INL}$ $[\mathrm{LSB}]$")
            ax2.set_xlabel(r"$\mathrm{Combined}$ $\mathrm{Code}$")
            ax2.grid(True, linestyle='--', alpha=0.3)
            ax2.set_xlim(0, max_output_length - 1)
            plt.tight_layout()
            plt.savefig(out_dir / "monte_carlo_inl.pdf", dpi=300)
            plt.close(fig2)
        return

    # =========================================================================
    # STANDARD SINGLE-TRACE PROCESSING BRANCH FALLBACK
    # =========================================================================
    else:
        print("\n[STANDARD MODE] Processing singular evaluation trajectory file dataset...")
        if args.coarse_file and args.fine_file:
            c_y = pd.to_numeric(df_coarse.iloc[:, 1], errors='coerce').dropna().values
            f_y = pd.to_numeric(df_fine.iloc[:, 1], errors='coerce').dropna().values

            num_total_codes = 8192
            target_fine_len = num_total_codes // len(c_y)
            xp = np.linspace(0, 1, len(f_y))
            x_new = np.linspace(0, 1, target_fine_len)
            f_y_resampled = np.interp(x_new, xp, f_y)

            total_delay = []
            for c_val in c_y:
                for f_val in f_y_resampled: total_delay.append(c_val + f_val)
            y = np.array(total_delay)[:num_total_codes]
        else:
            df = pd.read_csv(args.csv)
            df.columns = [c.strip() for c in df.columns]
            y = pd.to_numeric(df[df.columns[1]], errors="coerce").dropna().to_numpy(dtype=float)

        blocks = split_into_coarse_blocks(y, coarse_codes=args.coarse_codes, fine_codes=args.fine_codes)
        delay, coarse_marker, coarse_codes_out, fine_codes_out, selected_input_indices, info = combine_like_coarse_fine_dtc(
            blocks, coarse_codes=args.coarse_codes, fine_codes=args.fine_codes,
            max_boundary_skip=args.max_boundary_skip, remove_coarse=args.remove_coarse,
            remove_fine=args.remove_fine, slope_negative=args.slope_negative,
        )

        dnl, inl, lsb = compute_dnl_inl(delay)
        
        with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
            delay_plot, dnl_plot, inl_plot = plot_results(delay, coarse_marker, dnl, inl, out_dir)

        combined_path = out_dir / "processed_combined_delay_like_coarse_fine_dtc.json"
        info_path = out_dir / "processed_info_like_coarse_fine_dtc.json"
        coarse_counts_json_path = out_dir / "processed_coarse_code_counts_like_coarse_fine_dtc.json"
        coarse_counts_csv_path = out_dir / "processed_coarse_code_counts_like_coarse_fine_dtc.csv"
        avg_power_csv_path = out_dir / "processed_avg_power_selected_like_coarse_fine_dtc.csv"

        unique_coarse, coarse_counts = np.unique(coarse_codes_out, return_counts=True)
        coarse_count_map = {int(c): int(n) for c, n in zip(unique_coarse, coarse_counts)}
        first_fine_per_coarse = {}
        last_fine_per_coarse = {}
        for c, f in zip(coarse_codes_out, fine_codes_out):
            c_i = int(c)
            f_i = int(f)
            if c_i not in first_fine_per_coarse: first_fine_per_coarse[c_i] = f_i
            last_fine_per_coarse[c_i] = f_i

        with combined_path.open("w", encoding="utf-8") as f: json.dump(delay.tolist(), f, indent=2)
        with info_path.open("w", encoding="utf-8") as f: json.dump({"policy": info, "n_input_points": len(y)}, f, indent=2)
        with coarse_counts_json_path.open("w", encoding="utf-8") as f: json.dump({"coarse_code_counts": coarse_count_map}, f, indent=2)
        pd.DataFrame({"coarse_code": unique_coarse.astype(int), "n_codes": coarse_counts.astype(int)}).to_csv(coarse_counts_csv_path, index=False)

        if has_power_files:
            print("\nSynthesizing total processed average power matching active selection codes...")
            p_c_vals = pd.to_numeric(df_p_coarse.iloc[:, 1], errors='coerce').dropna().values[:args.coarse_codes]
            p_f_vals = pd.to_numeric(df_p_fine.iloc[:, 1], errors='coerce').dropna().values[:args.fine_codes]

            target_fine_len_p = 8192 // len(p_c_vals)
            xp_p = np.linspace(0, 1, len(p_f_vals))
            x_new_p = np.linspace(0, 1, target_fine_len_p)
            p_f_resampled = np.interp(x_new_p, xp_p, p_f_vals)

            power_list = []
            for c_hw, f_hw in zip(coarse_codes_out, fine_codes_out):
                c_idx = int(c_hw) if int(c_hw) < len(p_c_vals) else -1
                f_idx = int(f_hw) if int(f_hw) < len(p_f_resampled) else -1
                power_list.append(p_c_vals[c_idx] + p_f_resampled[f_idx])
            
            avg_power_selected = np.array(power_list) * 1e-6
            valid_selected = np.arange(len(delay), dtype=int)
            selected_coarse_codes = coarse_codes_out
            selected_fine_codes = fine_codes_out
        else:
            power_csv_list = [Path(p) for p in args.power_csvs]
            if len(power_csv_list) > 0:
                avg_power_all = compute_average_power_per_period(
                    power_csv_list=power_csv_list, n_periods=len(y),
                    period_s=float(args.period_s), start_time_s=args.power_start_time
                )
                valid_selected = selected_input_indices[(selected_input_indices >= 0) & (selected_input_indices < len(avg_power_all))]
                avg_power_selected = avg_power_all[valid_selected]
                selected_coarse_codes = coarse_codes_out[: len(valid_selected)].astype(int)
                selected_fine_codes = fine_codes_out[: len(valid_selected)].astype(int)

                (selected_coarse_codes, selected_fine_codes, valid_selected, avg_power_selected) = filter_power_codes_for_plot(
                    coarse_codes=selected_coarse_codes, fine_codes=selected_fine_codes,
                    source_period_index=valid_selected.astype(int), avg_power_w=avg_power_selected,
                )
            else:
                avg_power_selected = np.zeros(len(delay))
                valid_selected = np.arange(len(delay))
                selected_coarse_codes = coarse_codes_out
                selected_fine_codes = fine_codes_out

        linear_code = np.arange(len(avg_power_selected), dtype=int)
        pd.DataFrame({
            "linearized_code": linear_code,
            "source_period_index": valid_selected.astype(int),
            "coarse_code": selected_coarse_codes,
            "fine_code": selected_fine_codes,
            "avg_power_w": avg_power_selected,
            "avg_power_uw": avg_power_selected * 1e6,
        }).to_csv(avg_power_csv_path, index=False)

        with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
            avg_power_plot = plot_selected_average_power(avg_power_selected, out_dir, P_static=args.static_power_uw * 1e-6)

        print(f"Single evaluation complete. Peak |DNL|: {float(np.max(np.abs(dnl))):.4f}")
if __name__ == "__main__":
    main()

