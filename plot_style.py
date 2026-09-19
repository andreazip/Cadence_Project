from __future__ import annotations

import shutil
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def _detect_working_latex() -> bool:
    """Check that matplotlib's usetex pipeline actually works end to end.

    `latex` can be on PATH and compile a plain document while still failing
    matplotlib's own preamble, e.g. when the `cm-super`/`type1cm` packages it
    requires are missing (common with a bare BasicTeX/conda install). So the
    only reliable check is to run matplotlib's real TeX pipeline, not a
    hand-written probe document.
    """
    if shutil.which("latex") is None:
        return False
    try:
        from matplotlib.texmanager import TexManager

        with plt.rc_context({"text.usetex": True}):
            TexManager().get_text_width_height_descent("test", 10, None)
        return True
    except Exception:
        return False


HAS_SCIENCEPLOTS = False
HAS_LATEX = _detect_working_latex()
print(f"Detected working LaTeX: {HAS_LATEX}")
SCIENCE_STYLE = ["science", "std-colors"]
SCIENCE_STYLE_OVERRIDES = {
    # Falls back to matplotlib's built-in mathtext when no LaTeX install is found,
    # so labels like $\mathrm{...}$ still render instead of crashing plt.tight_layout().
    "text.usetex": HAS_LATEX,
    "figure.figsize": (3.3, 2.5),
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
}
SHOW_FIGURE_TITLES = False


try:
    import scienceplots  # noqa: F401

    HAS_SCIENCEPLOTS = True
    plt.rcdefaults()
    plt.style.use(SCIENCE_STYLE)
except ImportError:
    HAS_SCIENCEPLOTS = False


def apply_science_style() -> None:
    """Apply the standard plotting style globally."""
    if HAS_SCIENCEPLOTS:
        plt.style.use(SCIENCE_STYLE)
    plt.rcParams.update(SCIENCE_STYLE_OVERRIDES)


def with_science_style(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if HAS_SCIENCEPLOTS:
            with plt.style.context(SCIENCE_STYLE):
                with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
                    return func(*args, **kwargs)
        with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
            return func(*args, **kwargs)

    return wrapper


def maybe_title(target, text, **kwargs):
    if SHOW_FIGURE_TITLES and _has_multiple_plot_axes(target.figure):
        target.set_title(text, **kwargs)


def maybe_suptitle(fig, text, **kwargs):
    if SHOW_FIGURE_TITLES and _has_multiple_plot_axes(fig):
        fig.suptitle(text, **kwargs)


def _has_multiple_plot_axes(fig):
    plot_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    return len(plot_axes) > 1


def _style_axis(ax, xbins=6):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", width=0.5)
    ax.tick_params(axis="x", which="minor", width=0.5)


def _multi_panel_figsize(nrows, ncols):
    base_w, base_h = plt.rcParams.get("figure.figsize", (6.4, 4.8))
    width_scale = max(1, ncols)
    height_scale = max(1, nrows)
    if ncols == 2:
        height_scale *= 1.5  # slightly reduce height for 2-column layouts
    elif ncols >= 3:
        height_scale *= 1.7  # more reduction for 3+ columns
    return base_w * width_scale, base_h * height_scale


def _save_png_and_pdf(fig, png_path, dpi=300, bbox_inches="tight"):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)

    pdf_dir = png_path.parent / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{png_path.stem}.pdf"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)
