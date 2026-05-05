from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dtc.dtc_core import ConstantSlopeDTC, DelayLineDTC, VariableSlopeDTC, compute_sigma_c

# ===== PUBLICATION-READY PLOT STYLE (matching DTC_simulation.py) =====
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "lines.linewidth": 2.6,
    "lines.markersize": 4,
    "lines.markeredgewidth": 1.0,
    "grid.alpha": 0.6,
    "grid.color": "#b7b7b7",
    "grid.linestyle": "--",
    "grid.linewidth": 1.2,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 1.6,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "xtick.major.width": 1.4,
    "xtick.minor.width": 1.0,
    "ytick.major.width": 1.4,
    "ytick.minor.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
})


@dataclass
class DTCConfig:
    """Configuration for one DTC block."""

    n: int = 8
    Cu: float = 8e-15
    Vdd: float = 1.1
    f: float = 20e6
    Ich: float = 350e-9
    Cramp: float = 5e-15
    # Optional delay-line ramp capacitance. If None, Cramp is used.
    Cramp_dl: Optional[float] = None
    # Optional VS CDAC unit capacitance. If None, Cramp is used.
    C_ramp_cu: Optional[float] = None
    Vth: Optional[float] = None

    C_array: Optional[np.ndarray] = None
    C0: Optional[float] = None
    C0_scale: float = 1

    enable_CLM: bool = False
    enable_nonlin: bool = False
    C1: float = 0.323
    C2: float = -0.09
    I1: float = 0.184

    C_fixed: float = 0.0

    # DAC coding mode for the LSB capacitor network: 'binary' or 'thermometer'
    dac_mode: str = "binary"

    # DTC slope type: 'constant' (CS), 'variable' (VS), or 'delay_line' (DL)
    slope_mode: str = "constant"

    # Delay-line output selection behavior:
    # - 'tapped': all stages are active; output tap is selected (constant power vs code)
    # - 'accumulated': active stage count grows with code (power rises with code)
    delay_line_selection_mode: str = "tapped"

    # Mode-dependent voltage scaling factors applied to both Vdd and Vth.
    vdd_vth_factor_constant: float = 2.0 / 3.0
    vdd_vth_factor_variable: float = 0.5

    # Optional extra power reduction factor (divide by 4 when enabled), split by mode.
    # If these are not provided, legacy `self_power_down` is used as fallback.
    self_power_down_vs: Optional[Union[bool, str]] = None
    self_power_down_dl: Optional[Union[bool, str]] = None
    # Legacy unified switch kept for backward compatibility.
    self_power_down: Union[bool, str] = True

    # Mismatch parameters: sigma(Cu)/Cu = Ac/sqrt(A)
    Ac: float = 5.218e-3
    A: float = 4.33


@dataclass
class DTCModel:
    """Single DTC model: digital code -> delay and power."""

    config: DTCConfig

    def __post_init__(self) -> None:
        if self.config.Vth is None:
            self.config.Vth = self.config.Vdd / 2.0

        # Delay-line ramp capacitance defaults to Cramp unless explicitly provided.
        self._cramp_dl = self.config.Cramp if self.config.Cramp_dl is None else self.config.Cramp_dl

        
        # VS CDAC unit capacitance defaults to Cramp unless explicitly provided.
        self._vs_cdac_cu = self.config.Cramp if self.config.C_ramp_cu is None else self.config.C_ramp_cu

        self._dac_mode = self.config.dac_mode.strip().lower()
        if self._dac_mode not in {"binary", "thermometer"}:
            raise ValueError("dac_mode must be 'binary' or 'thermometer'")

        self._slope_mode = self.config.slope_mode.strip().lower()
        if self._slope_mode not in {"constant", "variable", "delay_line"}:
            raise ValueError("slope_mode must be 'constant', 'variable', or 'delay_line'")

        self._delay_line_selection_mode = str(self.config.delay_line_selection_mode).strip().lower()
        if self._delay_line_selection_mode not in {"tapped", "accumulated"}:
            raise ValueError("delay_line_selection_mode must be 'tapped' or 'accumulated'")

        # Mode-dependent supply scaling (configurable from runner-level CONFIG).
        self._supply_factor = (
            self.config.vdd_vth_factor_variable
            if self._slope_mode in {"variable", "delay_line"}
            else self.config.vdd_vth_factor_constant
        )
        self._vdd_eff = self.config.Vdd * self._supply_factor
        self._vth_eff = float(self.config.Vth) * self._supply_factor

        def _to_bool(value: Union[bool, str]) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"yes", "true", "1", "on"}
            return bool(value)

        spd_fallback = self.config.self_power_down
        spd_vs = self.config.self_power_down_vs
        spd_dl = self.config.self_power_down_dl

        self._self_power_down_vs = _to_bool(spd_vs if spd_vs is not None else spd_fallback)
        self._self_power_down_dl = _to_bool(spd_dl if spd_dl is not None else spd_fallback)

        self._run_flags = {
            "CLM": self.config.enable_CLM,
            "Non-linearities-capacitor": self.config.enable_nonlin,
        }

        if self.config.C_array is None:
            if self._slope_mode == "constant":
                bank_bits = self.config.n - 1
                unit_cap = self.config.Cu
            elif self._slope_mode == "variable":
                # VS uses one full DAC bank with n-bit code space.
                bank_bits = self.config.n
                unit_cap = self._vs_cdac_cu
            else:
                # Delay-line mode does not require DAC capacitors for code mapping.
                bank_bits = 0
                unit_cap = self._cramp_dl

            if self._slope_mode == "delay_line":
                self._c_array = np.array([unit_cap], dtype=float)
            elif self._dac_mode == "binary":
                self._c_array = np.array(
                    [(2**j) * unit_cap for j in range(bank_bits)],
                    dtype=float,
                )
            else:
                n_units = (2**bank_bits) - 1
                self._c_array = np.full(n_units, unit_cap, dtype=float)
        else:
            self._c_array = np.array(self.config.C_array, dtype=float)

        self._ca = float(np.sum(self._c_array))
        self._c0 = float(self.config.C0) if self.config.C0 is not None else self.config.C0_scale * self._ca

        sigma_c_cs = compute_sigma_c(self.config.Ac, self.config.A, self.config.Cu)
        sigma_c_vs = compute_sigma_c(self.config.Ac, self.config.A, self._vs_cdac_cu)
        self._cs_core = ConstantSlopeDTC(
            n_bits=self.config.n,
            cu=self.config.Cu,
            dac_mode=self._dac_mode,
            thermo_bits=4,
            run_flags=self._run_flags,
            sigma_c=sigma_c_cs,
            vdd=self._vdd_eff,
            vth=self._vth_eff,
            ich=self.config.Ich,
            cramp=self.config.Cramp,
            i1=self.config.I1,
            c1=self.config.C1,
            c2=self.config.C2,
        )
        # VS core uses one full n-bit DAC bank.
        self._vs_core = VariableSlopeDTC(
            n_bits=self.config.n,
            dac_mode=self._dac_mode,
            thermo_bits=4,
            run_flags=self._run_flags,
            sigma_c=sigma_c_vs,
            vdd=self._vdd_eff,
            vth=self._vth_eff,
            ich=self.config.Ich,
            cramp_u=self._vs_cdac_cu,
            C_fixed = self.config.C_fixed,
            i1=self.config.I1,
            c1=self.config.C1,
            c2=self.config.C2,
        )

        # Align code-space with dtc_core convention: N = 2^n - 1.
        self.n_codes = (2 ** self.config.n) - 1

    def active_codes(self) -> np.ndarray:
        """Return active code indices following dtc behavior per slope mode.

        - Constant slope (CS): remove midpoint code.
        - Variable slope (VS): keep all codes.
        - Delay-line (DL): keep all codes.
        """
        codes = np.arange(self.n_codes, dtype=int)
        if self._slope_mode == "constant":
            mid = (len(codes) - 1) // 2
            return np.delete(codes, mid)
        return codes

    def _calc_ck_constant_slope(self, lsb_code: int) -> float:
        return float(self._cs_core.calc_ck(lsb_code, self._c_array))

    def _calc_ck_variable_slope(self, code: int) -> float:
        return float(self._vs_core.calc_ck(code, self._c_array))

    def _compute_vst_cs(self, msb: int, c_k: float) -> float:
        if msb == 1:
            return (1 + (self._ca - c_k) / (self._c0 + self.config.Cramp + self._ca)) * self._vdd_eff
        return (1 - c_k / (self._c0 + self.config.Cramp + self._ca)) * self._vdd_eff

    def _compute_energy_cs(self, msb: int, c_k: float, vst: float) -> float:
        if msb == 1:
            return float(self._cs_core.energy_msb_1(self._ca, c_k, self._c0, vst))
        return float(self._cs_core.energy_msb_0(c_k, self._c0, self._ca))

    def evaluate(self, code: int) -> Tuple[float, float]:
        """Return (delay_s, power_w) for a single digital code."""
        if code < 0 or code >= self.n_codes:
            raise ValueError(f"Code must be in [0, {self.n_codes - 1}]")

        lsb_code = code & ((2 ** (self.config.n - 1)) - 1)
        msb = (code >> (self.config.n - 1)) & 1

        if self._slope_mode == "constant":
            c_k = self._calc_ck_constant_slope(lsb_code)
            vst = self._compute_vst_cs(msb, c_k)
            energy = self._compute_energy_cs(msb, c_k, vst)
            cramp_nom = self.config.Cramp
        elif self._slope_mode == "variable":
            # VS mode: one n-bit DAC controls effective ramp capacitance directly.
            c_k = self._calc_ck_variable_slope(code)
            # Use normalized code-dependent level for optional CLM/nonlinearity perturbation.
            vst = (c_k / max(self._ca, 1e-30)) * self._vdd_eff
            energy = c_k * self._vdd_eff**2 + self.config.C_fixed * self._vdd_eff**2
            cramp_nom = c_k
        else:
            # Delay-line mode: code controls number of enabled replicas.
            n_rep = float(code + 1)
            vst = 0.0
            cramp_nom = self._cramp_dl
            if self._delay_line_selection_mode == "tapped":
                # Tapped line: all delay elements stay active; selected output tap changes delay only.
                energy = cramp_nom * self._vdd_eff**2 * float(self.n_codes)
            else:
                # Accumulated line: number of active elements grows with code.
                energy = cramp_nom * self._vdd_eff**2 * n_rep

        ich_eff = self.config.Ich
        cramp_eff = cramp_nom

        if self._slope_mode != "delay_line" and self.config.enable_CLM:
            ich_eff = self.config.Ich * (1 + self.config.I1 * (vst - 0.4))

        if self._slope_mode != "delay_line" and self.config.enable_nonlin:
            cramp_eff = cramp_nom * (1 + self.config.C1 * vst + self.config.C2 * vst**2)

        if ich_eff == 0:
            raise ValueError("Ich effective value is zero; cannot compute delay")

        if self._slope_mode == "constant":
            k_eff = -ich_eff / cramp_eff
            delay = (self._vth_eff - vst) / k_eff
        elif self._slope_mode == "variable":
            delay = cramp_eff * (self._vdd_eff - self._vth_eff) / ich_eff
        else:
            n_rep = float(code + 1)
            delay = cramp_eff * (self._vdd_eff - self._vth_eff) / ich_eff * n_rep
        power = energy * self.config.f

        if self._slope_mode == "variable" and self._self_power_down_vs:
            power = power / 4.0
        elif self._slope_mode == "delay_line" and self._self_power_down_dl:
            power = power / 4.0

        return float(delay), float(power)

    def characterize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays of delays and powers over all codes."""
        delays = np.zeros(self.n_codes)
        powers = np.zeros(self.n_codes)
        for code in range(self.n_codes):
            delays[code], powers[code] = self.evaluate(code)
        return delays, powers

    def with_cap_mismatch(self, rng: Optional[np.random.Generator] = None) -> "DTCModel":
        """
        Return a new DTCModel with capacitor mismatch sampled from area-based sigma.

        sigma_c = Ac/sqrt(A) * Cu and each binary-weighted capacitor is sampled as:
        Cj = (2**j) * (Cu + N(0, sigma_c)).
        """
        if rng is None:
            rng = np.random.default_rng()

        if self._slope_mode == "constant":
            bank_bits = self.config.n - 1
            unit_cap = self.config.Cu
        elif self._slope_mode == "variable":
            bank_bits = self.config.n
            unit_cap = self._vs_cdac_cu
        else:
            sigma_c = compute_sigma_c(self.config.Ac, self.config.A, self._cramp_dl)
            cfg_dict = dict(self.config.__dict__)
            cfg_dict['Cramp_dl'] = max(1e-30, float(self._cramp_dl + rng.normal(0.0, sigma_c)))
            return DTCModel(DTCConfig(**cfg_dict))

        sigma_c = compute_sigma_c(self.config.Ac, self.config.A, unit_cap)

        if self._dac_mode == "binary":
            c_array_mc = np.zeros(bank_bits, dtype=float)
            for j in range(bank_bits):
                c_array_mc[j] = (2**j) * (unit_cap + rng.normal(0.0, sigma_c/np.sqrt(2**j)))
        else:
            n_units = (2**bank_bits) - 1
            c_array_mc = np.zeros(n_units, dtype=float)
            for j in range(n_units):
                c_array_mc[j] = unit_cap + rng.normal(0.0, sigma_c)

        cfg_dict = dict(self.config.__dict__)
        cfg_dict['C_array'] = c_array_mc
        if self.config.C0 is None:
            # Match DTC_simulation behavior: keep C0 fixed during MC sweeps.
            cfg_dict['C0'] = float(self._c0)

        return DTCModel(DTCConfig(**cfg_dict))


class DTCModelCS(DTCModel):
    """Convenience class for Constant-Slope DTC."""

    def __init__(self, config: DTCConfig):
        cfg_dict = dict(config.__dict__)
        cfg_dict["slope_mode"] = "constant"
        super().__init__(DTCConfig(**cfg_dict))


class DTCModelVS(DTCModel):
    """Convenience class for Variable-Slope DTC."""

    def __init__(self, config: DTCConfig):
        cfg_dict = dict(config.__dict__)
        cfg_dict["slope_mode"] = "variable"
        super().__init__(DTCConfig(**cfg_dict))


class DTCModelDL(DTCModel):
    """Convenience class for Delay-Line DTC mode."""

    def __init__(self, config: DTCConfig):
        cfg_dict = dict(config.__dict__)
        cfg_dict["slope_mode"] = "delay_line"
        super().__init__(DTCConfig(**cfg_dict))


@dataclass
class CoarseFineDTC:
    """Two-stage coarse-fine architecture using two DTC blocks."""

    coarse: DTCModel
    fine: DTCModel

    coarse_delays: np.ndarray = field(init=False)
    coarse_powers: np.ndarray = field(init=False)
    fine_delays: np.ndarray = field(init=False)
    fine_powers: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.coarse_delays, self.coarse_powers = self.coarse.characterize()
        self.fine_delays, self.fine_powers = self.fine.characterize()

    def _active_coarse_codes(self) -> np.ndarray:
        return self.coarse.active_codes()

    def _active_fine_codes(self) -> np.ndarray:
        return self.fine.active_codes()

    def _estimate_coarse_period(self) -> float:
        """Estimate coarse period from adjacent coarse-code delay differences."""
        coarse_codes = self._active_coarse_codes()
        coarse_delays_active = self.coarse_delays[coarse_codes]
        d = np.abs(np.diff(coarse_delays_active))
        d = d[d > 0]
        if len(d) == 0:
            raise ValueError("Unable to estimate coarse period from coarse delays")
        return float(np.median(d))

    def _valid_fine_indices(self, coarse_period_s: Optional[float]) -> np.ndarray:
        """Return fine-code indices that fit inside one coarse period."""
        if coarse_period_s is None:
            coarse_period_s = self._estimate_coarse_period()

        fine_codes = self._active_fine_codes()
        fine_delays_active = self.fine_delays[fine_codes]
        fine_min = float(np.min(fine_delays_active))
        fine_span = fine_delays_active - fine_min
        valid_local = np.where((fine_span >= 0.0) & (fine_span <= coarse_period_s))[0]
        valid = fine_codes[valid_local]
        if len(valid) == 0:
            raise ValueError("No fine codes fall inside one coarse period")
        return valid

    def _fine_indices_by_transition_policy(self) -> Tuple[np.ndarray, list[np.ndarray], Dict[str, float]]:
        """
        Build per-coarse selected fine indices using transition policy:

        - First coarse segment uses all active fine codes.
                - For each next segment, start at first fine point whose total delay is above
                    previous selected point plus 0.5 * local_lsb, where local_lsb is mean
          absolute fine step in that segment.
        """
        coarse_codes = self._active_coarse_codes()
        fine_idx_all = self._active_fine_codes()

        selected: list[np.ndarray] = []
        boundary_skip_count = 0
        boundary_violation_count = 0
        boundary_no_solution_count = 0
        prev_idx = np.array(fine_idx_all, dtype=int)
        fine_idx = np.array(fine_idx_all, dtype=int)
        
        start_local = 0
        end_previous = 0
        prev_vals = + self.coarse_delays[0] + self.fine_delays[fine_idx]
        
        lsb = np.mean(np.abs(np.diff(prev_vals))) if len(prev_vals) > 1 else 0.0

        coarse_codes = coarse_codes[1:]
       
        for c_code in coarse_codes:
        
            candidate_local = np.array([], dtype=int)
            local_vals = self.coarse_delays[c_code] + self.fine_delays[fine_idx]  
        
            iterate = True

            if c_code > 0 and len(local_vals) > 0:
                for (i,p) in enumerate(prev_vals):
                    if iterate:
                        p = float(p)
                        margin = local_vals - p
                        
                        candidate = np.where((margin >= 0.5 * lsb) & (margin <= 1.5 * lsb))[0]
                        if len(candidate) > 0 and len(candidate_local) == 0:
            
                                end_previous = i
                                candidate_local = candidate
                                iterate = False

                # if c_code == 1:
                #     print(f"Coarse code {c_code}: candidate local indices {candidate_local}, lsb {lsb:.3e}, margin {margin[candidate_local]}, start {candidate_local[0] if len(candidate_local) > 0 else 'N/A'}, end{end_previous if len(candidate_local) > 0 else 'N/A'}")
                #     print(f"prev_vals: {prev_vals[end_previous]}, start_next {local_vals[candidate_local[0]] if len(candidate_local) > 0 else 'N/A'}, margin = {prev_vals[end_previous] - local_vals[candidate_local[0]] if len(candidate_local) > 0 else 'N/A'}")
                
                if iterate:
                    prev_value = prev_vals[-1]
                    end_previous = len(prev_vals) - 1
                    start_local = 0
                    boundary_no_solution_count += 1
                    candidate_local = local_vals - (prev_value + lsb)
                    start_local = np.argmin(np.abs(candidate_local))
                else:    
                    start_local = int(candidate_local[0]) 


            boundary_skip_count += start_local
            fine_idx = fine_idx[start_local:]
            local_vals = local_vals[start_local:]
            prev_vals = prev_vals[:end_previous+1]
            prev_idx = prev_idx[:end_previous+1]


            selected.append(prev_idx)
            prev_idx = fine_idx
            prev_vals = local_vals

        selected.append(prev_idx)
    
        meta = {
            "boundary_skip_count": float(boundary_skip_count),
            "boundary_violation_count": float(boundary_violation_count),
            "boundary_no_solution_count": float(boundary_no_solution_count),
        }
        return coarse_codes, selected, meta

    def combined_characteristic(self, coarse_period_s: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Return coarse-fine characteristic with rollover at one coarse period.

        For each coarse code, fine codes inside one coarse period are used,
        except for the last coarse segment where the full fine range is allowed.
        """
        coarse_codes, fine_idx_per_coarse, policy_meta = self._fine_indices_by_transition_policy()

        combined_codes = []
        total_delays = []
        total_powers = []
        coarse_codes_out = []
        fine_codes_out = []
        coarse_markers = []

        code_counter = 0
        coarse_codes = self._active_coarse_codes()

        for c_local, c_code in enumerate(coarse_codes):
            fine_idx = fine_idx_per_coarse[c_local]
            for local_i, f_code in enumerate(fine_idx):
                combined_codes.append(code_counter)
                total_delays.append(self.coarse_delays[c_code] + self.fine_delays[f_code])
                total_powers.append(self.coarse_powers[c_code] + self.fine_powers[f_code])
                coarse_codes_out.append(c_code)
                fine_codes_out.append(int(f_code))
                coarse_markers.append(local_i == 0)
                code_counter += 1

        return {
            'combined_code': np.array(combined_codes, dtype=int),
            'total_delay_s': np.array(total_delays, dtype=float),
            'total_power_w': np.array(total_powers, dtype=float),
            'coarse_code': np.array(coarse_codes_out, dtype=int),
            'fine_code': np.array(fine_codes_out, dtype=int),
            'coarse_marker': np.array(coarse_markers, dtype=bool),
            'policy_boundary_skip_count': np.array([policy_meta['boundary_skip_count']], dtype=float),
            'policy_boundary_violation_count': np.array([policy_meta['boundary_violation_count']], dtype=float),
        }

    
    def synthesize_delay(self, target_delay_s: float, coarse_period_s: Optional[float] = None) -> Dict[str, float]:
        """
        Solve target delay with only coarse + fine DTC codes.

        The solver checks all coarse/fine code combinations and returns the
        pair that minimizes absolute delay error.
        """
        if target_delay_s < 0:
            raise ValueError("target_delay_s must be >= 0")

        coarse_idx, fine_idx_per_coarse, _ = self._fine_indices_by_transition_policy()

        candidates = []
        for coarse_local, coarse_code in enumerate(coarse_idx):
            fine_idx = fine_idx_per_coarse[coarse_local]
            for fine_code in fine_idx:
                total_delay = self.coarse_delays[coarse_code] + self.fine_delays[fine_code]
                candidates.append((int(coarse_code), int(fine_code), float(total_delay)))

        best = min(candidates, key=lambda x: abs(x[2] - target_delay_s))
        coarse_code, fine_code, total_delay = best

        coarse_delay = float(self.coarse_delays[coarse_code])
        fine_delay = float(self.fine_delays[fine_code])

        p_coarse = float(self.coarse_powers[coarse_code])
        p_fine = float(self.fine_powers[fine_code])

        return {
            "target_delay_s": float(target_delay_s),
            "coarse_code": float(coarse_code),
            "coarse_delay_s": coarse_delay,
            "fine_code": float(fine_code),
            "fine_delay_s": fine_delay,
            "total_delay_s": float(total_delay),
            "error_s": float(total_delay - target_delay_s),
            "p_coarse_w": p_coarse,
            "p_fine_w": p_fine,
            "p_total_w": float(p_coarse + p_fine),
        }

    def sweep_target_delays(self, target_delays_s: np.ndarray, coarse_period_s: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Evaluate coarse+fine synthesis on a vector of target delays."""
        targets = np.array(target_delays_s, dtype=float)

        total_delays = np.zeros_like(targets)
        errors = np.zeros_like(targets)
        p_totals = np.zeros_like(targets)
        coarse_codes = np.zeros_like(targets, dtype=int)
        fine_codes = np.zeros_like(targets, dtype=int)

        for i, t in enumerate(targets):
            r = self.synthesize_delay(float(t), coarse_period_s=coarse_period_s)
            total_delays[i] = r['total_delay_s']
            errors[i] = r['error_s']
            p_totals[i] = r['p_total_w']
            coarse_codes[i] = int(r['coarse_code'])
            fine_codes[i] = int(r['fine_code'])

        return {
            'target_delay_s': targets,
            'total_delay_s': total_delays,
            'error_s': errors,
            'p_total_w': p_totals,
            'coarse_code': coarse_codes,
            'fine_code': fine_codes,
        }

    def plot_total_power(
        self,
        delay_range_ns: Optional[float] = None,
        num_points: int = 300,
        coarse_period_s: Optional[float] = None,
        save_path: Optional[str] = None,
    ):
        """Plot total power (coarse + fine) versus realizable delay.

        This avoids artificial endpoint clamping that appears when sweeping
        unreachable target delays and mapping with nearest-code synthesis.
        """
        ch = self.combined_characteristic(coarse_period_s=coarse_period_s)
        delays_ns = ch['total_delay_s'] * 1e9
        powers_uw = ch['total_power_w'] * 1e6

        order = np.argsort(delays_ns)
        delays_ns = delays_ns[order]
        powers_uw = powers_uw[order]

        if delay_range_ns is not None:
            keep = delays_ns <= delay_range_ns
            delays_ns = delays_ns[keep]
            powers_uw = powers_uw[keep]

        if len(delays_ns) == 0:
            raise ValueError("No realizable delay points in requested delay_range_ns")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(delays_ns, powers_uw, color='#1F77B4', linewidth=2.2)
        ax.set_title('Coarse-Fine DTC Total Power vs Target Delay', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Target Delay [ns]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Power [uW]', fontsize=12, fontweight='bold')
        ax.set_xlim(float(np.min(delays_ns)), float(np.max(delays_ns)))
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax

    def plot_characteristic_vs_code(
        self,
        t_range_ns: Optional[float] = None,
        coarse_period_s: Optional[float] = None,
        save_path: Optional[str] = None,
    ):
        """Plot coarse-fine characteristic (total delay vs combined code)."""
        ch = self.combined_characteristic(coarse_period_s=coarse_period_s)
        codes = ch['combined_code']
        delays_ns = ch['total_delay_s'] * 1e9
        coarse_marker = ch['coarse_marker']

        if t_range_ns is not None:
            keep = delays_ns <= t_range_ns
            codes = codes[keep]
            delays_ns = delays_ns[keep]
            coarse_marker = coarse_marker[keep]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(codes, delays_ns, color='#D62728', linewidth=2.6, label='Coarse-Fine Delay Characteristic')
        ax.plot(
            codes[coarse_marker],
            delays_ns[coarse_marker],
            'o',
            color='#1F77B4',
            markersize=7,
            markeredgewidth=1.2,
            markeredgecolor='white',
            label='Coarse Transition',
            zorder=5,
        )
        ax.set_title('Coarse-Fine Delay Characteristic vs Combined Code', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Combined Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
        if t_range_ns is not None:
            ax.set_ylim(0, t_range_ns)
        else:
            ax.set_ylim(0, float(np.max(delays_ns)))
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax

    def plot_power_vs_code(
        self,
        coarse_period_s: Optional[float] = None,
        avg_over_target_range_ns: Optional[float] = None,
        avg_num_points: int = 200,
        save_path: Optional[str] = None,
    ):
        """Plot total power vs combined code over full non-redundant code range."""
        ch = self.combined_characteristic(coarse_period_s=coarse_period_s)
        codes = ch['combined_code']
        powers_uw = ch['total_power_w'] * 1e6

        if avg_over_target_range_ns is not None:
            target_delays_s = np.linspace(0.0, float(avg_over_target_range_ns) * 1e-9, int(avg_num_points))
            sweep = self.sweep_target_delays(target_delays_s, coarse_period_s=coarse_period_s)
            p_avg_uw = float(np.mean(sweep['p_total_w']) * 1e6)
        else:
            p_avg_uw = float(np.mean(powers_uw)) if len(powers_uw) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(
            codes,
            powers_uw,
            color='#1F77B4',
            linewidth=2.6,
            label=f'Coarse-Fine Total Power (P_avg = {p_avg_uw:.3f} uW)',
        )
        ax.set_title('Coarse-Fine Total Power vs Combined Code', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Combined Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Power [uW]', fontsize=12, fontweight='bold')
        if len(codes) > 0:
            ax.set_xlim(0, int(codes[-1]))
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax

    def plot_single_block_characteristic(
        self,
        block: str,
        t_range_ns: Optional[float] = None,
        save_path: Optional[str] = None,
    ):
        """Plot delay characteristic vs code for coarse-only or fine-only block."""
        block_key = block.strip().lower()
        if block_key not in {"coarse", "fine"}:
            raise ValueError("block must be 'coarse' or 'fine'")

        if block_key == "coarse":
            delays = self.coarse_delays
            active_codes = self._active_coarse_codes()
            color = '#D62728'
            title = 'Coarse DTC Delay Characteristic vs Code'
            label = 'Coarse Delay'
        else:
            delays = self.fine_delays
            active_codes = self._active_fine_codes()
            color = '#1F77B4'
            title = 'Fine DTC Delay Characteristic vs Code'
            label = 'Fine Delay'

        delay_ns = delays[active_codes] * 1e9
        code_axis = np.array(active_codes, dtype=int)

        if t_range_ns is not None:
            keep = delay_ns <= t_range_ns
            code_axis = code_axis[keep]
            delay_ns = delay_ns[keep]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(code_axis, delay_ns, color=color, linewidth=2.6, label=label)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delay [ns]', fontsize=12, fontweight='bold')
        if t_range_ns is not None:
            ax.set_ylim(0, t_range_ns)
        else:
            ax.set_ylim(0, float(np.max(delay_ns)))
        if len(code_axis) > 0:
            ax.set_xlim(int(code_axis[0]), int(code_axis[-1]))
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax

    def plot_single_block_nonlinearity(
        self,
        block: str,
        mc_runs: int = 100,
        dnl_limit_lsb: float = 0.5,
        save_path: Optional[str] = None,
    ):
        """Plot Monte Carlo DNL/INL for coarse-only or fine-only block."""
        block_key = block.strip().lower()
        if block_key not in {"coarse", "fine"}:
            raise ValueError("block must be 'coarse' or 'fine'")

        rng = np.random.default_rng()

        if block_key == "coarse":
            base_model = self.coarse
            active_codes = self._active_coarse_codes()
            title_prefix = 'Coarse DTC'
        else:
            base_model = self.fine
            active_codes = self._active_fine_codes()
            title_prefix = 'Fine DTC'

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        dnl_peaks = []
        lsb_last = None
        pass_count = 0

        for mc in range(mc_runs):
            mc_model = base_model.with_cap_mismatch(rng)
            delays, _ = mc_model.characterize()
            delay_active = delays[active_codes]

            if len(delay_active) < 3:
                raise ValueError("Not enough points to compute DNL/INL")

            lsb = (delay_active[-1] - delay_active[0]) / (len(delay_active) - 1)
            lsb_last = lsb
            dnl = np.diff(delay_active) / lsb - 1
            # Zero-referenced INL so INL starts at 0 at the first code.
            inl = np.concatenate(([0.0], np.cumsum(dnl)))
            # Map DNL step error to lower code edge; INL is point-wise on active codes.
            codes_dnl = active_codes[:-1]
            codes_inl = active_codes

            peak = float(np.max(np.abs(dnl)))
            dnl_peaks.append(peak)
            if peak < dnl_limit_lsb:
                pass_count += 1

            alpha_value = 0.15 + (mc / max(mc_runs - 1, 1)) * 0.5
            ax1.plot(codes_dnl, dnl, linewidth=1.6, alpha=alpha_value, color='#D62728')
            ax2.plot(codes_inl, inl, linewidth=1.6, alpha=alpha_value, color='#1F77B4')

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax1.set_ylabel('DNL (LSB)', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'{title_prefix} Monte Carlo Non-Linearity ({mc_runs} runs, LSB = {lsb_last:.2e})',
            fontsize=14,
            fontweight='bold',
            pad=12,
        )
        ax1.axhline(y=dnl_limit_lsb, color='red', linestyle='--', linewidth=1.0, alpha=0.4,
                    label=f'±{dnl_limit_lsb:.1f} LSB')
        ax1.axhline(y=-dnl_limit_lsb, color='red', linestyle='--', linewidth=1.0, alpha=0.4)
        ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax2.set_ylabel('INL (LSB)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax2.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        pass_prob = (pass_count / mc_runs) * 100.0
        print(f"{title_prefix} probability of staying below {dnl_limit_lsb:.1f} LSB = {pass_prob:.2f}%")

        return fig, (ax1, ax2)

    def plot_coarse_nonlinearity_ps(
        self,
        mc_runs: int = 100,
        save_path: Optional[str] = None,
    ):
        """Plot coarse-only Monte Carlo DNL/INL with y-axis in ps error."""
        rng = np.random.default_rng()
        active_codes = self._active_coarse_codes()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

        for mc in range(mc_runs):
            mc_model = self.coarse.with_cap_mismatch(rng)
            delays, _ = mc_model.characterize()
            delay_active = delays[active_codes]

            if len(delay_active) < 3:
                raise ValueError("Not enough points to compute DNL/INL")

            lsb_s = (delay_active[-1] - delay_active[0]) / (len(delay_active) - 1)
            dnl_ps = (np.diff(delay_active) - lsb_s) * 1e12
            # Zero-referenced INL in ps so it starts at 0 at first code.
            inl_ps = np.concatenate(([0.0], np.cumsum(dnl_ps)))
            # Map DNL step error to lower code edge; INL is point-wise on active codes.
            codes_dnl = active_codes[:-1]
            codes_inl = active_codes

            alpha_value = 0.15 + (mc / max(mc_runs - 1, 1)) * 0.5
            ax1.plot(codes_dnl, dnl_ps, linewidth=1.6, alpha=alpha_value, color='#D62728')
            ax2.plot(codes_inl, inl_ps, linewidth=1.6, alpha=alpha_value, color='#1F77B4')

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax1.set_ylabel('DNL Error [ps]', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'Coarse DTC Monte Carlo Non-Linearity ({mc_runs} runs, Error in ps)',
            fontsize=14,
            fontweight='bold',
            pad=12,
        )
        ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax1.set_axisbelow(True)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax2.set_ylabel('INL Error [ps]', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Digital Code', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax2.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, (ax1, ax2)

    def measure_phase_noise(
        self,
        code: int,
        num_samples: int = 2**16,
        nperseg: int = 1024,
        temperature_k: float = 100e-3,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate phase noise L(f) for one combined coarse-fine code.

        Thermal-jitter model:
        sigma_rj = sqrt(k*T*C_tot)/Ich_fine
        where C_tot = Cramp + Cu*(2**n_fine)
        """
        try:
            from scipy.signal import welch
        except ImportError as exc:
            raise ImportError("scipy is required for phase-noise estimation (welch)") from exc

        ch = self.combined_characteristic()
        if code < 0 or code >= len(ch['combined_code']):
            raise ValueError(f"code must be in [0, {len(ch['combined_code']) - 1}]")

        fine_code = int(ch['fine_code'][code])
        _, _ = self.fine.evaluate(fine_code)

        k_b = 1.38e-23
        c_tot = self.fine.config.Cramp + self.fine.config.Cu * (2**self.fine.config.n)
        if self.fine.config.Ich <= 0:
            raise ValueError("fine Ich must be > 0")

        sigma_rj = np.sqrt(k_b * temperature_k * c_tot) / self.fine.config.Ich

        rng = np.random.default_rng(seed)
        jitter_samples = rng.normal(0.0, sigma_rj, int(num_samples))

        f_ref = self.coarse.config.f
        phase_error = 2 * np.pi * f_ref * jitter_samples

        nperseg_eff = min(int(nperseg), int(num_samples))
        freqs, psd = welch(phase_error, fs=f_ref, nperseg=nperseg_eff)

        # L(f) in dBc/Hz
        phase_noise_db = 10.0 * np.log10(np.maximum(psd / 2.0, 1e-300))
        return freqs, phase_noise_db

    def plot_phase_noise(
        self,
        codes: Union[int, Sequence[int]],
        num_samples: int = 2**16,
        nperseg: int = 1024,
        temperature_k: float = 100e-3,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """Plot phase noise L(f) for one or multiple combined codes."""
        if isinstance(codes, int):
            code_list = [codes]
        else:
            code_list = [int(c) for c in codes]

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#8C564B']

        for idx, code in enumerate(code_list):
            local_seed = None if seed is None else seed + idx
            freqs, phase_noise_db = self.measure_phase_noise(
                code=code,
                num_samples=num_samples,
                nperseg=nperseg,
                temperature_k=temperature_k,
                seed=local_seed,
            )
            valid = freqs > 0
            ax.semilogx(
                freqs[valid],
                phase_noise_db[valid],
                color=colors[idx % len(colors)],
                linewidth=2.2,
                label=f'Code {code}',
            )

        ax.set_title('Estimated Phase Noise L(f)', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Offset Frequency [Hz]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Phase Noise [dBc/Hz]', fontsize=12, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, framealpha=0.96, edgecolor='black', loc='best')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig, ax

    def max_sigma_jitter_tolerated_from_lf(
        self,
        l_dbc_hz: float,
        integration_bw_hz: float,
    ) -> float:
        """
        Convert flat phase-noise target L(f) to max tolerated sigma_jitter [s].

        Assumes single-sideband flat L(f) over integration bandwidth B:
        sigma_phi^2 = 2 * 10^(L/10) * B
        sigma_t = sigma_phi / (2*pi*f_ref)
        """
        if integration_bw_hz <= 0:
            raise ValueError("integration_bw_hz must be > 0")

        f_ref = self.coarse.config.f
        s_phi = 2.0 * (10.0 ** (l_dbc_hz / 10.0))
        sigma_phi = np.sqrt(s_phi * integration_bw_hz)
        sigma_t = sigma_phi / (2.0 * np.pi * f_ref)
        return float(sigma_t)


def build_coarse_fine_dtc(
    coarse_values: Dict,
    fine_values: Dict,
    coarse_model: str = "auto",
    fine_model: str = "auto",
) -> CoarseFineDTC:
    """
    Convenience constructor from dictionaries.

    coarse_values and fine_values use DTCConfig fields.
    coarse_model/fine_model can be: 'auto', 'cs', 'vs', or 'dl'.
    """
    coarse_cfg = DTCConfig(**coarse_values)
    fine_cfg = DTCConfig(**fine_values)

    def make_model(cfg: DTCConfig, model_sel: str) -> DTCModel:
        sel = model_sel.strip().lower()
        if sel == "auto":
            mode = cfg.slope_mode.strip().lower()
            if mode == "variable":
                return DTCModelVS(cfg)
            if mode == "delay_line":
                return DTCModelDL(cfg)
            return DTCModelCS(cfg)
        if sel == "cs":
            return DTCModelCS(cfg)
        if sel == "vs":
            return DTCModelVS(cfg)
        if sel == "dl":
            return DTCModelDL(cfg)
        raise ValueError("model selector must be 'auto', 'cs', 'vs', or 'dl'")

    return CoarseFineDTC(
        coarse=make_model(coarse_cfg, coarse_model),
        fine=make_model(fine_cfg, fine_model),
    )


def run_mc_mismatch_analysis(
    coarse_values: Dict,
    fine_values: Dict,
    mc_runs: int = 200,
    delay_range_ns: float = 10.0,
    num_points: int = 150,
    dnl_limit_lsb: float = 0.5,
    t_range_ns: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes], Dict[str, float]]:
    """
    Monte Carlo mismatch analysis (DNL/INL cloud) for coarse-fine DTC.

    Each Monte Carlo run samples capacitor mismatch for both coarse and fine DTCs
    using area-based sigma, then computes DNL and INL over combined code.
    """
    rng = np.random.default_rng()
    coarse_nom = DTCModel(DTCConfig(**coarse_values))
    fine_nom = DTCModel(DTCConfig(**fine_values))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    dnl_peaks = []
    passing_runs = 0
    lsb_last = None

    for mc in range(mc_runs):
        arch_mc = CoarseFineDTC(
            coarse=coarse_nom.with_cap_mismatch(rng),
            fine=fine_nom.with_cap_mismatch(rng),
        )
        ch = arch_mc.combined_characteristic()
        delays_s = ch['total_delay_s']
        codes = ch['combined_code']

        if t_range_ns is not None:
            keep = delays_s * 1e9 <= t_range_ns
            delays_s = delays_s[keep]
            codes = codes[keep]

        if len(delays_s) < 3:
            raise ValueError("Not enough code points after t_range_ns filtering for DNL/INL calculation")

        lsb = (delays_s[-1] - delays_s[0]) / (len(delays_s) - 1)
        lsb_last = lsb
        dnl = np.diff(delays_s) / lsb - 1
        # Zero-referenced INL so INL starts at 0 at first combined code.
        inl = np.concatenate(([0.0], np.cumsum(dnl)))

        dnl_peak = float(np.max(np.abs(dnl)))
        dnl_peaks.append(dnl_peak)
        if dnl_peak < dnl_limit_lsb:
            passing_runs += 1

        codes_dnl = codes[:-1]
        codes_inl = codes

        alpha_value = 0.15 + (mc / max(mc_runs - 1, 1)) * 0.5
        ax1.plot(codes_dnl, dnl, alpha=alpha_value, linewidth=1.6, color='#D62728')
        ax2.plot(codes_inl, inl, alpha=alpha_value, linewidth=1.6, color='#1F77B4')

    ax1.set_ylabel("DNL (LSB)", fontsize=12, fontweight='bold')
    ax1.set_title(
        f"Coarse-Fine Monte Carlo DNL/INL ({mc_runs} Realizations, LSB = {lsb_last:.2e})",
        fontsize=14,
        fontweight='bold',
        pad=12,
    )
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax1.axhline(y=dnl_limit_lsb, color='red', linestyle='--', linewidth=1.0, alpha=0.4,
                label=f'±{dnl_limit_lsb:.1f} LSB')
    ax1.axhline(y=-dnl_limit_lsb, color='red', linestyle='--', linewidth=1.0, alpha=0.4)
    ax1.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax1.set_axisbelow(True)

    ax2.set_ylabel("INL (LSB)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Combined Code", fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.2, color="#b7b7b7")
    ax2.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    coarse_sigma_c = coarse_nom.config.Ac / np.sqrt(coarse_nom.config.A) * coarse_nom.config.Cu
    fine_sigma_c = fine_nom.config.Ac / np.sqrt(fine_nom.config.A) * fine_nom.config.Cu

    stats = {
        'mc_runs': float(mc_runs),
        'coarse_sigma_c_f': float(coarse_sigma_c),
        'fine_sigma_c_f': float(fine_sigma_c),
        'lsb_s': float(lsb_last),
        'pass_probability_percent': float((passing_runs / mc_runs) * 100.0),
        'max_dnl_peak_lsb': float(np.max(dnl_peaks)),
        'mean_dnl_peak_lsb': float(np.mean(dnl_peaks)),
    }

    return fig, (ax1, ax2), stats

def calculate_current(td, J =0.8e-12, F = 0.05, q =1.6e-19):
        "Calculate current based on jitter requirements considering shot noise."
        return F*q*td/J**2

def optimize_split_loop(
    n_total: int,
    base_coarse_values: Dict,
    base_fine_values: Dict,
    n_coarse_values: Optional[np.ndarray] = None,
    max_delay_ns: float = 5.0,
    mc_runs: int = 100,
    dnl_limit_lsb: float = 0.5,
    num_points_power: int = 250,
    use_same_t_range_as_main: bool = True,
) -> Tuple[list, Dict]:
    """
    Sweep n_coarse to find a coarse/fine split for fixed n_total.

    Rules:
    n_fine = n_total - n_coarse
    Ich_fine = Vdd/(5e-12 * 2**n_fine - 1) * Cramp

    Stored metrics per split:
    - mean total power over target-delay sweep
    - probability of staying below dnl_limit_lsb for whole system
    """
    if n_coarse_values is None:
        n_coarse_values = np.arange(2, n_total - 1)  # keep both blocks with enough points

    results = []
    utilization = 0.9

    def _coarse_dac_cap_sum(values: Dict, n_bits_coarse: int) -> float:
        """Return total CS DAC capacitance seen by the coarse block."""
        c_array = values.get('C_array')
        if c_array is not None:
            return float(np.sum(np.array(c_array, dtype=float)))

        bank_bits = max(int(n_bits_coarse) - 1, 0)
        if bank_bits == 0:
            return 0.0

        cu = float(values['Cu'])
        n_units = (2**bank_bits) - 1
        return float(n_units * cu)

    for n_coarse in n_coarse_values:
        n_fine = int(n_total - n_coarse)
        if n_fine < 2:
            continue

        coarse_values = dict(base_coarse_values)
        fine_values = dict(base_fine_values)

        coarse_values['n'] = int(n_coarse)
        fine_values['n'] = int(n_fine)

        coarse_mode = str(coarse_values.get('slope_mode', 'constant')).strip().lower()
        coarse_is_vs = coarse_mode == 'variable'
        coarse_is_dl = coarse_mode == 'delay_line'
        coarse_vdd= coarse_values['Vdd'] 
        if coarse_is_vs:
            # VS coarse scaling: use full n-bit code space (2^n - 1 steps).
            res_coarse = 5e-9 / (2**n_coarse - 1)
            coarse_vth = coarse_values.get('Vth', coarse_values['Vdd'] / 2)
            coarse_values['Vth'] = coarse_vth
            k_slope_coarse = (coarse_vdd - coarse_vth) / res_coarse
            coarse_values['Ich'] = calculate_current(5e-9)
            coarse_values['C_ramp_cu'] = coarse_values['Ich'] / k_slope_coarse
            if coarse_values['C_ramp_cu'] < 2e-15:
                coarse_values['C_ramp_cu'] = 2e-15 #for linearity
                coarse_values['Ich'] = coarse_values['C_ramp_cu'] * k_slope_coarse #it will be greater than minimum value
            N_fine = 2**n_fine - 1
        elif coarse_is_dl:
            # Delay-line coarse scaling: delay step per replica is set by Cramp.
            res_coarse = 5e-9 / (2**n_coarse - 1)
            coarse_vth = coarse_values.get('Vth', coarse_values['Vdd'] / 2)
            coarse_values['Vth'] = coarse_vth
            k_slope_coarse = (coarse_vdd - coarse_vth) / res_coarse
            coarse_values['Ich'] = calculate_current(5e-9)
            coarse_values['Cramp_dl'] = coarse_values['Ich']/ k_slope_coarse
            if coarse_values['Cramp_dl'] < 2e-15:
                coarse_values['Cramp_dl'] = 2e-15 #for linearity
                coarse_values['Ich'] = coarse_values['Cramp_dl'] * k_slope_coarse #it will be greater than minimum value
            
            N_fine = 2**n_fine - 1
        else:
            coarse_codes = 2**n_coarse-2 #target number of coarse codes, can be adjusted as needed
            N_fine = 2**n_fine - 2 

            # User-defined coarse current scaling law for CS coarse block.
            res_coarse = 5e-9/coarse_codes #the coarse resolution is given by the target range divided by the target number of coarse codes
            k_slope_coarse = (coarse_vdd / (res_coarse * ((2**n_coarse) - 2)))

            coarse_values['Ich'] = calculate_current(5e-9)
            C_ramp = coarse_values['Ich'] / k_slope_coarse  
            coarse_values['Cramp'] = C_ramp

            c_sum = _coarse_dac_cap_sum(coarse_values, n_coarse)
            cu_step = float(coarse_values.get('cu_increment', 1e-15))
            if cu_step <= 0:
                raise ValueError("cu_increment must be > 0 when enforcing Cramp/Csum rule")

            # Enforce: if Cramp < Csum set C0 = Csum - Cramp; else increase Cu by one step.
            if C_ramp >= c_sum:
                if coarse_values.get('C_array') is not None:
                    c_array = np.array(coarse_values['C_array'], dtype=float)
                    while C_ramp >= c_sum:
                        c_array = c_array + cu_step
                        c_sum = float(np.sum(c_array))
                    coarse_values['C_array'] = c_array
                    coarse_values['Cu'] = float(np.min(c_array))
                else:
                    while C_ramp >= c_sum:
                        coarse_values['Cu'] = float(coarse_values['Cu']) + cu_step
                        c_sum = _coarse_dac_cap_sum(coarse_values, n_coarse)

            coarse_values['C0'] = float(c_sum - C_ramp)

        fine_mode = str(fine_values.get('slope_mode', 'constant')).strip().lower()
        fine_is_vs = fine_mode == 'variable'
        fine_is_dl = fine_mode == 'delay_line'
        fine_vdd = fine_values['Vdd'] 
        if fine_is_vs:
            # VS fine scaling: use full n-bit code space (2^n - 1 steps).
            fine_codes = int(N_fine* utilization)
            res_fine = res_coarse / fine_codes #the fine resolution is given by the coarse resolution divided by the target number of fine codes
            
            fine_vth = fine_values.get('Vth', fine_values['Vdd'] / 2)
            fine_values['Vth'] = fine_vth
            
            k_slope_fine = (fine_vdd- fine_vth) / res_fine
            fine_values['Ich'] = calculate_current(res_fine, J = 0.5e-12)
            fine_values['C_ramp_cu'] = fine_values['Ich'] / k_slope_fine

            if fine_values['C_ramp_cu'] < 1e-15:
                fine_values['C_ramp_cu'] = 1e-15 #for linearity
                fine_values['Ich'] = fine_values['C_ramp_cu'] * k_slope_fine #it will be greater than minimum value
            
            
        elif fine_is_dl:
            # Delay-line fine scaling: use replica-step delay set by Cramp.
            fine_codes = int(N_fine * utilization)
            res_fine = res_coarse / fine_codes

            fine_vth = fine_values.get('Vth', fine_values['Vdd'] / 2)
            fine_values['Vth'] = fine_vth

            k_slope_fine = (fine_vdd - fine_vth) / res_fine
            fine_values['Ich'] = calculate_current(res_fine, J = 0.5e-12)
            fine_values['Cramp_dl'] = fine_values['Ich'] / k_slope_fine

            if fine_values['Cramp_dl'] < 1e-15:
                fine_values['Cramp_dl'] = 1e-15 #for linearity
                fine_values['Ich'] = fine_values['Cramp_dl'] * k_slope_fine #it will be greater than minimum value
            

        else:
            fine_codes = int(N_fine * utilization) #target number of fine codes, can be adjusted as needed

            # User-defined fine current scaling law for CS fine block.
            res_fine = res_coarse/fine_codes #the fine resolution is given by the coarse resolution divided by the target number of fine codes
            k_slope_fine = (fine_vdd / (res_fine * ((2**n_fine) - 2)))

            fine_values['Ich'] = calculate_current(res_coarse, J = 0.5e-12)
            C_ramp = fine_values['Ich'] / k_slope_fine

            if C_ramp < 1e-15:
                C_ramp = 1e-15 #for linearity
                fine_values['Ich'] = C_ramp * k_slope_fine #it will be greater than minimum value
                
            fine_values['Cramp'] = C_ramp


           
            c_sum = _coarse_dac_cap_sum(fine_values, n_fine)
            cu_step = float(coarse_values.get('cu_increment', 1e-15))
            if cu_step <= 0:
                raise ValueError("cu_increment must be > 0 when enforcing Cramp/Csum rule")

            # Enforce: if Cramp < Csum set C0 = Csum - Cramp; else increase Cu by one step.
            if C_ramp >= c_sum:
                if fine_values.get('C_array') is not None:
                    c_array = np.array(fine_values['C_array'], dtype=float)
                    while C_ramp >= c_sum:
                        c_array = c_array + cu_step
                        c_sum = float(np.sum(c_array))
                    fine_values['C_array'] = c_array
                    fine_values['Cu'] = float(np.min(c_array))
                else:
                    while C_ramp >= c_sum:
                        fine_values['Cu'] = float(fine_values['Cu']) + cu_step
                        c_sum = _coarse_dac_cap_sum(fine_values, n_fine)

            fine_values['C0'] = float(c_sum - C_ramp)

        architecture = build_coarse_fine_dtc(coarse_values, fine_values)

        target_delays_s = np.linspace(0.0, max_delay_ns * 1e-9, num_points_power)
        sweep = architecture.sweep_target_delays(target_delays_s)
        avg_power_w = float(np.mean(sweep['p_total_w']))
        max_power_w = float(np.max(sweep['p_total_w']))

        try:
            fig_mc, _, mc_stats = run_mc_mismatch_analysis(
                coarse_values=coarse_values,
                fine_values=fine_values,
                mc_runs=mc_runs,
                dnl_limit_lsb=dnl_limit_lsb,
                # For apples-to-apples comparison with main simulation, use the
                # same delay window by default.
                t_range_ns=max_delay_ns if use_same_t_range_as_main else None,
                save_path=None,
            )
            plt.close(fig_mc)
            pass_prob = float(mc_stats['pass_probability_percent'])
        except ValueError as err:
            if 'Not enough code points' not in str(err):
                raise
            pass_prob = 0.0

        results.append({
            'n_total': int(n_total),
            'n_coarse': int(n_coarse),
            'n_fine': int(n_fine),
            'coarse_mode': coarse_mode,
            'fine_mode': fine_mode,
            'ich_coarse_a': float(coarse_values['Ich']),
            'ich_fine_a': float(fine_values['Ich']),

            # Coarse final params (all exported; display is mode-dependent in runner).
            'cu_coarse_f': float(coarse_values.get('Cu', np.nan)),
            'cramp_coarse_f': float(coarse_values.get('Cramp', np.nan)),
            'c0_coarse_f': float(coarse_values.get('C0')) if coarse_values.get('C0') is not None else np.nan,
            'cramp_cu_coarse_f': float(coarse_values.get('C_ramp_cu', np.nan)),
            'cramp_dl_coarse_f': float(coarse_values.get('Cramp_dl', np.nan)),

            # Fine final params (all exported; display is mode-dependent in runner).
            'cu_fine_f': float(fine_values.get('Cu', np.nan)),
            'cramp_fine_f': float(fine_values.get('Cramp', np.nan)),
            'c0_fine_f': float(fine_values.get('C0')) if fine_values.get('C0') is not None else np.nan,
            'cramp_cu_fine_f': float(fine_values.get('C_ramp_cu', np.nan)),
            'cramp_dl_fine_f': float(fine_values.get('Cramp_dl', np.nan)),

            'avg_total_power_w': avg_power_w,
            'max_total_power_w': max_power_w,
            'pass_probability_percent': pass_prob,
        })

    if len(results) == 0:
        raise ValueError('No valid split evaluated')

    # Prioritize P_max first, then pass %, then P_avg.
    # Use table-equivalent rounding to avoid tiny float noise deciding ties.
    best = sorted(
        results,
        key=lambda r: (
            round(r['max_total_power_w'] * 1e6, 3),
            -round(r['pass_probability_percent'], 2),
            round(r['avg_total_power_w'] * 1e6, 3),
        ),
    )[0]
    return results, best

