"""
trajectory_plot.py — publication-quality rendering of a MatchTrajectory. All styling lives
here; changing colors, fonts, or layout never requires touching trajectory_generation.py or
trajectory_events.py (requirement 10's separation of concerns).

Design: data-journalism aesthetic in the Tennis Abstract / FiveThirtyEight lineage — a
serif display face for editorial weight, a muted navy/terracotta pair (colorblind-safe,
deliberately not default red/blue), thin dotted set-boundary lines with labels above the
plot rather than cluttering the data area, and small leader-line callouts for events.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator

from tennis_intel.viz.trajectory_generation import MatchTrajectory
from tennis_intel.viz.trajectory_events import (
    SetBoundary, MatchEvent, detect_set_boundaries, detect_events,
)

# --- Palette: colorblind-safe, deliberately not default red/blue ---
COLOR_P1 = "#1B4B6B"       # deep navy
COLOR_P2 = "#C15A3E"       # muted terracotta
COLOR_OUTCOME = "#2B2B2B"  # near-black, for the actual-result reference line
COLOR_GRID = "#E4E1D8"
COLOR_SET_LINE = "#9A9689"
COLOR_BG = "#FBFAF7"       # warm off-white, not stark white
COLOR_EVENT = "#6B6456"

EVENT_MARKER_STYLE = {
    "break": {"marker": "o", "size": 5, "color": "#C15A3E"},
    "tiebreak_start": {"marker": "s", "size": 5, "color": "#6B6456"},
    "tiebreak_end": {"marker": "s", "size": 5, "color": "#6B6456"},
    "match_point": {"marker": "^", "size": 6, "color": "#1B4B6B"},
    "championship_point": {"marker": "*", "size": 9, "color": "#B8860B"},
}


def _setup_fonts() -> tuple[str, str]:
    """Serif for display/title (editorial weight), sans for body/axes (data clarity).
    Falls back gracefully if the preferred faces aren't installed in this environment."""
    serif_candidates = ["Georgia", "Palatino", "DejaVu Serif"]
    sans_candidates = ["Helvetica", "Arial", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    serif = next((f for f in serif_candidates if f in available), "DejaVu Serif")
    sans = next((f for f in sans_candidates if f in available), "DejaVu Sans")
    return serif, sans


def plot_trajectory(
    traj: MatchTrajectory,
    match_df_for_events,  # the same per-point dataframe used to build the trajectory
    out_path: str | Path,
    show_confidence_band: bool = False,
    confidence_lower: list[float] | None = None,
    confidence_upper: list[float] | None = None,
) -> Path:
    """
    Renders the full publication-quality figure and saves it at 300 DPI.

    SIMPLIFIED TO ML-INFORMED MARKOV ONLY (2026-07): per project direction — ML-Informed
    Markov is the primary, actively-optimized engine going forward (a historically-
    grounded XGBoost pre-match baseline, updated coherently point by point via the
    Beta-Binomial mechanism) — this chart now shows ONLY that engine, for both players,
    rather than all four previously plotted (Markov, ML+MC, ML-Informed Markov, Hybrid).
    Markov/ML+MC/Hybrid are still computed and stored on MatchTrajectory (build_trajectory
    is unchanged) since that's cheap and harmless, but are no longer drawn here.
    """
    serif, sans = _setup_fonts()
    out_path = Path(out_path)

    if traj.ml_informed_p1 is None:
        raise ValueError(
            "traj.ml_informed_p1 is None — this chart now shows ONLY the ML-Informed "
            "Markov engine and requires that data. Make sure build_trajectory was called "
            "with pre_match_ml_informed_p1 set, and that the replay CSV has an "
            "'ml_informed_markov_p1' (or 'ml_informed_pred') column — re-run "
            "replay_match.py for this match if it predates that engine."
        )

    boundaries = detect_set_boundaries(match_df_for_events)
    events = detect_events(match_df_for_events)
    # Event/boundary point indices are relative to charted points (1..N); shift by +1 to
    # align with the trajectory's point_index, which reserves 0 for pre-match.
    for b in boundaries:
        b.point_index += 0  # already 1-indexed from charted data == trajectory index (no shift needed)
    for e in events:
        e.point_index += 0

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # --- Main trajectory lines: both players, mirrored (requirement 3) ---
    # ML-Informed Markov only — the project's primary engine (historically-grounded
    # XGBoost pre-match baseline, updated coherently point by point). Uses the same
    # prominent styling (solid, thick) Markov previously had as the chart's sole anchor.
    p1_probs = traj.ml_informed_p1
    p2_probs = [1 - p for p in traj.ml_informed_p1]

    ax.plot(traj.point_index, p1_probs, color=COLOR_P1, lw=2.2,
            label=f"{traj.p1_name} — ML-Informed Markov", zorder=5)
    ax.plot(traj.point_index, p2_probs, color=COLOR_P2, lw=2.2,
            label=f"{traj.p2_name} — ML-Informed Markov", zorder=5)

    if show_confidence_band and confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(traj.point_index, confidence_lower, confidence_upper,
                        color=COLOR_P1, alpha=0.12, zorder=1, label="Uncertainty band")

    # --- Pre-match and final points emphasized (requirements 1 and 2) ---
    ax.scatter([0], [p1_probs[0]], s=90, color=COLOR_P1, zorder=6,
              edgecolor=COLOR_BG, linewidth=1.5)
    ax.scatter([0], [p2_probs[0]], s=90, color=COLOR_P2, zorder=6,
              edgecolor=COLOR_BG, linewidth=1.5)
    final_idx = traj.point_index[-1]
    ax.scatter([final_idx], [p1_probs[-1]], s=140, color=COLOR_P1, zorder=7,
              edgecolor=COLOR_BG, linewidth=2, marker="o")
    ax.scatter([final_idx], [p2_probs[-1]], s=140, color=COLOR_P2, zorder=7,
              edgecolor=COLOR_BG, linewidth=2, marker="o")

    ax.annotate("Pre-match", xy=(0, p1_probs[0]), xytext=(8, 12),
               textcoords="offset points", fontsize=9, color=COLOR_EVENT, fontfamily=sans)
    winner_name = traj.p1_name if traj.winner_is_p1 else traj.p2_name
    ax.annotate(f"Final: {winner_name} wins", xy=(final_idx, p1_probs[-1]),
               xytext=(-10, 14 if traj.winner_is_p1 else -18), textcoords="offset points",
               fontsize=9, color=COLOR_EVENT, fontfamily=sans, ha="right", fontweight="bold")

    # --- Set boundaries (requirement 4) ---
    for b in boundaries:
        ax.axvline(b.point_index, color=COLOR_SET_LINE, lw=1, ls=(0, (2, 3)), zorder=2)
        ax.annotate(f"End Set {b.set_number}\n{b.score_str}", xy=(b.point_index, 1.0),
                   xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                   fontsize=8.5, color=COLOR_SET_LINE, fontfamily=sans, linespacing=1.3)

    # --- Event annotations (requirement 5) ---
    seen_labels_at = {}
    for e in events:
        style = EVENT_MARKER_STYLE.get(e.kind, {"marker": "o", "size": 5, "color": COLOR_EVENT})
        y = traj.ml_informed_p1[min(e.point_index, len(traj.ml_informed_p1) - 1)]
        ax.scatter([e.point_index], [y], s=style["size"] ** 2, marker=style["marker"],
                  color=style["color"], zorder=8, edgecolor=COLOR_BG, linewidth=0.8)
        # Only label championship/match points and tiebreak boundaries directly (breaks are
        # frequent enough that labeling every one would clutter the chart — the markers
        # alone communicate the pattern, per requirement 5's "small annotations suffice").
        # Annotations alternate above/below the point to avoid colliding with the x-axis.
        if e.kind in ("championship_point", "tiebreak_start", "match_point"):
            key = round(e.point_index / 15)
            if key not in seen_labels_at:
                above = y < 0.5  # if the point sits low, annotate upward, and vice versa
                y_offset = 18 if above else -20
                va = "bottom" if above else "top"
                ax.annotate(e.label, xy=(e.point_index, y), xytext=(0, y_offset),
                           textcoords="offset points", fontsize=7.5, color=style["color"],
                           fontfamily=sans, ha="center", va=va,
                           arrowprops=dict(arrowstyle="-", color=style["color"], lw=0.6, alpha=0.6))
                seen_labels_at[key] = True

    # --- Axes (requirement 6) ---
    ax.set_xlabel("Cumulative point number", fontsize=11, fontfamily=sans, color="#333333",
                  labelpad=10)
    ax.set_ylabel("Probability of winning match", fontsize=11, fontfamily=sans, color="#333333")
    ax.set_ylim(-0.08, 1.1)
    ax.set_xlim(-max(3, final_idx * 0.01), final_idx * 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    tick_step = 50 if final_idx > 200 else 25
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.tick_params(axis="both", labelsize=9, colors="#333333")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#B8B4A8")
    ax.grid(axis="y", color=COLOR_GRID, lw=0.8, zorder=0)
    ax.axhline(0.5, color="#B8B4A8", lw=0.8, ls=":", zorder=1)

    # --- Legend: placed in its own reserved band below the axes, never overlapping data ---
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
                       fontsize=8.5, frameon=False, prop={"family": sans},
                       columnspacing=1.6, handlelength=2.2)

    # --- Title and metadata (requirement 8) ---
    title = f"{traj.p1_name} vs {traj.p2_name}"
    fig.suptitle(title, fontsize=20, fontfamily=serif, fontweight="bold",
                y=1.01, color="#1A1A1A")
    ax.set_title("Live Match Win Probability", fontsize=12.5, fontfamily=sans,
                color="#555555", pad=16, loc="center")

    subtitle_parts = []
    if traj.tournament:
        subtitle_parts.append(traj.tournament)
    if traj.surface:
        subtitle_parts.append(f"{traj.surface} court")
    subtitle_parts.append(f"Best of {traj.best_of}")
    if traj.final_score:
        subtitle_parts.append(f"Final score: {traj.final_score}")
    subtitle = "  •  ".join(subtitle_parts)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9.5, fontfamily=sans,
             color="#777777")

    # Explicit rect reserves: top ~7% for title/subtitle, bottom ~14% for legend + x-label,
    # so nothing gets clipped by savefig regardless of figure size.
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.92))
    fig.savefig(out_path, dpi=300, facecolor=COLOR_BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out_path