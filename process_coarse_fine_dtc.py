import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import apply_science_style, _multi_panel_figsize, maybe_suptitle, maybe_title, _multi_panel_figsize


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
    ax.legend([P_static], title="Estimated static power", loc="upper left")
    maybe_title(ax, "Average Power per 20 ns Execution (Linearized Coarse-Fine Codes)")
    ax.set_xlabel("Linearized combined code")
    ax.set_ylabel("Average power [uW]")
    ax.grid(True, alpha=0.35)

    out = out_dir / "processed_avg_power_selected_like_coarse_fine_dtc.png"
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


def plot_results(delay: np.ndarray, coarse_marker: np.ndarray, dnl: np.ndarray, inl: np.ndarray, out_dir: Path):
    """Save delay characteristic and DNL/INL plots."""
    x = np.arange(len(delay))

    fig1, ax1 = plt.subplots(constrained_layout=True)
    ax1.plot(x, delay, linewidth=1.6, label="Delay")
    ax1.plot(x[coarse_marker], delay[coarse_marker], "o", markersize=4, label="Coarse transition")
    maybe_suptitle(ax1, "Delay Characteristic (Processed Like coarse_fine_dtc)")
    ax1.set_xlabel("Combined code")
    ax1.set_ylabel("Delay [s]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    p1 = out_dir / "processed_delay_characteristic_like_coarse_fine_dtc.png"
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)

    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=_multi_panel_figsize(2, 1), constrained_layout=True)
    ax2.plot(x, dnl, linewidth=1.4)
    ax2.axhline(0.0, linestyle="--", linewidth=1.0)
    maybe_suptitle(ax2, "DNL (Processed Like coarse_fine_dtc)")
    ax2.set_xlabel("Combined code")
    ax2.set_ylabel("DNL [LSB]")
    ax2.grid(True, alpha=0.3)

    ax3.plot(x, inl, linewidth=1.4)
    ax3.axhline(0.0, linestyle="--", linewidth=1.0)
    maybe_suptitle(ax3, "INL (Processed Like coarse_fine_dtc)")
    ax3.set_xlabel("Combined code")
    ax3.set_ylabel("INL [LSB]")
    ax3.grid(True, alpha=0.3)

    p2 = out_dir / "processed_dnl_inl_like_coarse_fine_dtc.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)

    return p1, p2


def main():
    parser = argparse.ArgumentParser(description="Process CSV exactly like coarse_fine_dtc indexing/policy.")
    parser.add_argument("--csv", default="results_cadence/coarse_fine_dtc_delay.csv")
    parser.add_argument(
        "--power-csvs",
        nargs="+",
        default=["results_cadence/power_coarse_fine/power_coarse_fine_13bits_1.csv",
        "results_cadence/power_coarse_fine/power_coarse_fine_13bits_2.csv", "results_cadence/power_coarse_fine/power_coarse_fine_13bits_3.csv", "results_cadence/power_coarse_fine/power_coarse_fine_13bits_4.csv",    "results_cadence/power_coarse_fine/power_coarse_fine_13bits_5.csv", "results_cadence/power_coarse_fine/power_coarse_fine_13bits_6.csv" , "results_cadence/power_coarse_fine/power_coarse_fine_13bits_7.csv", "results_cadence/power_coarse_fine/power_coarse_fine_13bits_8.csv", "results_cadence/power_coarse_fine/power_coarse_fine_13bits_9.csv", "results_cadence/power_coarse_fine/power_coarse_fine_13bits_10.csv"],
        help="Ordered power transient CSVs; traces are processed sequentially in this order.",
    )
    parser.add_argument("--out-dir", default=str(SAVE_DIR))
    parser.add_argument("--coarse-codes", type=int, default=32)
    parser.add_argument("--fine-codes", type=int, default=64)
    parser.add_argument("--period-s" , type=float, default=20e-9)
    parser.add_argument("--power-start-time", type=float, default=None)
    parser.add_argument(
        "--max-boundary-skip",
        type=int,
        default=-1,
        help="Maximum skipped fine codes at each boundary (-1 means unlimited).",
    )
    parser.add_argument(
        "--remove-coarse", type=bool, default=False
    )
    parser.add_argument(
        "--remove-fine", type=bool, default=False
    )
    parser.add_argument(
        "--slope-negative", type=bool, default=False, help="Set to True if the delay characteristic is expected to decrease with increasing code."
    )
    parser.add_argument(
        "--static-power-uw", type=float, default=0.0, help="Estimated static power in microwatts to subtract from the average power plot.")
    
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    y = pd.to_numeric(df[df.columns[1]], errors="coerce").dropna().to_numpy(dtype=float)

    blocks = split_into_coarse_blocks(y, coarse_codes=args.coarse_codes, fine_codes=args.fine_codes)

    delay, coarse_marker, coarse_codes_out, fine_codes_out, selected_input_indices, info = combine_like_coarse_fine_dtc(
        blocks,
        coarse_codes=args.coarse_codes,
        fine_codes=args.fine_codes,
        max_boundary_skip=args.max_boundary_skip,
        remove_coarse=args.remove_coarse,
        remove_fine=args.remove_fine,
        slope_negative=args.slope_negative,
    )

    dnl, inl, lsb = compute_dnl_inl(delay)
    delay_plot, dnl_inl_plot = plot_results(delay, coarse_marker, dnl, inl, out_dir)

    combined_path = out_dir / "processed_combined_delay_like_coarse_fine_dtc.json"
    info_path = out_dir / "processed_info_like_coarse_fine_dtc.json"
    coarse_counts_json_path = out_dir / "processed_coarse_code_counts_like_coarse_fine_dtc.json"
    coarse_counts_csv_path = out_dir / "processed_coarse_code_counts_like_coarse_fine_dtc.csv"
    avg_power_csv_path = out_dir / "processed_avg_power_selected_like_coarse_fine_dtc.csv"

    unique_coarse, coarse_counts = np.unique(coarse_codes_out, return_counts=True)
    coarse_count_map = {int(c): int(n) for c, n in zip(unique_coarse, coarse_counts)}
    first_fine_per_coarse: dict[int, int] = {}
    last_fine_per_coarse: dict[int, int] = {}
    for c, f in zip(coarse_codes_out, fine_codes_out):
        c_i = int(c)
        f_i = int(f)
        if c_i not in first_fine_per_coarse:
            first_fine_per_coarse[c_i] = f_i
        last_fine_per_coarse[c_i] = f_i

    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(delay.tolist(), f, indent=2)

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": info,
                "n_input_points": int(len(y)),
                "n_blocks": int(len(blocks)),
                "n_output_points": int(len(delay)),
                "total_codes": int(len(delay)),
                "delay_min_s": float(np.min(delay)) if len(delay) > 0 else 0.0,
                "delay_max_s": float(np.max(delay)) if len(delay) > 0 else 0.0,
                "delay_range_s": float(np.max(delay) - np.min(delay)) if len(delay) > 0 else 0.0,
                "lsb_ideal": float(lsb),
                "dnl_peak_abs": float(np.max(np.abs(dnl))) if len(dnl) else 0.0,
                "inl_peak_abs": float(np.max(np.abs(inl))) if len(inl) else 0.0,
                "coarse_code_used": coarse_codes_out.tolist(),
                "fine_code_used": fine_codes_out.tolist(),
                "coarse_code_counts": coarse_count_map,
            },
            f,
            indent=2,
        )

    with coarse_counts_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "coarse_code_counts": coarse_count_map,
                "total_codes": int(len(delay)),
                "delay_min_s": float(np.min(delay)) if len(delay) > 0 else 0.0,
                "delay_max_s": float(np.max(delay)) if len(delay) > 0 else 0.0,
                "delay_range_s": float(np.max(delay) - np.min(delay)) if len(delay) > 0 else 0.0,
            },
            f,
            indent=2,
        )

    pd.DataFrame(
        {
            "coarse_code": unique_coarse.astype(int),
            "n_codes": coarse_counts.astype(int),
            "first_fine_code": [first_fine_per_coarse.get(int(c), -1) for c in unique_coarse],
            "last_fine_code": [last_fine_per_coarse.get(int(c), -1) for c in unique_coarse],
        }
    ).to_csv(coarse_counts_csv_path, index=False)

    power_csv_list = [Path(p) for p in args.power_csvs]
    avg_power_all = compute_average_power_per_period(
        power_csv_list=power_csv_list,
        n_periods=len(y),
        period_s=float(args.period_s),
        start_time_s=args.power_start_time,
    )
    valid_selected = selected_input_indices[(selected_input_indices >= 0) & (selected_input_indices < len(avg_power_all))]
    avg_power_selected = avg_power_all[valid_selected]
    selected_coarse_codes = coarse_codes_out[: len(valid_selected)].astype(int)
    selected_fine_codes = fine_codes_out[: len(valid_selected)].astype(int)

    (
        selected_coarse_codes,
        selected_fine_codes,
        valid_selected,
        avg_power_selected,
    ) = filter_power_codes_for_plot(
        coarse_codes=selected_coarse_codes,
        fine_codes=selected_fine_codes,
        source_period_index=valid_selected.astype(int),
        avg_power_w=avg_power_selected,
    )

    linear_code = np.arange(len(avg_power_selected), dtype=int)

    pd.DataFrame(
        {
            "linearized_code": linear_code,
            "source_period_index": valid_selected.astype(int),
            "coarse_code": selected_coarse_codes,
            "fine_code": selected_fine_codes,
            "avg_power_w": avg_power_selected,
            "avg_power_uw": avg_power_selected * 1e6,
        }
    ).to_csv(avg_power_csv_path, index=False)

    avg_power_plot = plot_selected_average_power(avg_power_selected, out_dir, P_static=args.static_power_uw * 1e-6)

    print(f"Input points: {len(y)}")
    print(f"Blocks: {len(blocks)}")
    print(f"Output points: {len(delay)}")
    print(f"Boundary skips by direct rule: {info['boundary_skip_count']}")
    print(f"Max boundary skip setting: {info['max_boundary_skip']}")
    print(f"Boundary violations after policy: {info['boundary_violation_count']}")
    print(f"Mean local boundary LSB: {info['boundary_lsb_mean']}")
    print(f"Max boundary margin after policy: {info['boundary_margin_max']}")
    print(f"fine_regular_len: {info['fine_regular_len']}")
    print(f"coarse_active_len: {info['coarse_active_len']}")
    print(f"LSB ideal: {lsb}")
    print(f"Peak |DNL|: {float(np.max(np.abs(dnl))) if len(dnl) else 0.0}")
    print(f"Peak |INL|: {float(np.max(np.abs(inl))) if len(inl) else 0.0}")
    print(f"Selected codes for avg power: {len(avg_power_selected)}")
    print(f"Mean selected avg power [uW]: {float(np.nanmean(avg_power_selected) * 1e6):.6f}")
    print(f"Saved: {combined_path}")
    print(f"Saved: {info_path}")
    print(f"Saved: {coarse_counts_json_path}")
    print(f"Saved: {coarse_counts_csv_path}")
    print(f"Saved: {avg_power_csv_path}")
    print(f"Saved: {avg_power_plot}")
    print(f"Saved: {delay_plot}")
    print(f"Saved: {dnl_inl_plot}")


if __name__ == "__main__":
    main()
